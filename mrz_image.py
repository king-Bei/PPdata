"""Geometric MRZ candidates and line crops without passport labels or OCR guesses."""

import cv2
import numpy as np
from PIL import Image, ImageOps


def text_mask(image):
    """Suppress slowly varying illumination while retaining dark character strokes."""
    gray = np.asarray(image.convert('L'))
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 31, 15)


def split_mrz_lines(image):
    """Return the two strongest separated text bands, excluding narrow background marks."""
    mask = text_mask(image)
    height, width = mask.shape
    if height < 24 or width / height < 3:
        return []
    # Horizontal MRZ strokes cover much more of the width than isolated marks do.
    density = (mask > 0).sum(axis=1).astype(float)
    density = np.convolve(density, np.ones(3)/3, mode='same')
    active = density > max(width*.045, float(density.max())*.25)
    bands = []
    start = None
    for y, value in enumerate(np.r_[active, False]):
        if value and start is None:
            start = y
        elif not value and start is not None:
            if bands and start - bands[-1][1] <= 2:
                bands[-1] = (bands[-1][0], y)
            else:
                bands.append((start, y))
            start = None
    candidates = []
    for top, bottom in bands:
        if bottom-top < 6 or bottom-top > height*.55:
            continue
        columns = (mask[top:bottom] > 0).sum(axis=0)
        occupied = np.flatnonzero(columns > 0)
        if not len(occupied) or occupied[-1]-occupied[0] < width*.65:
            continue
        score = float(density[top:bottom].sum())
        candidates.append((score, top, bottom))
    pairs = []
    for i, (score1, top1, bottom1) in enumerate(candidates):
        for score2, top2, bottom2 in candidates[i+1:]:
            h1, h2 = bottom1-top1, bottom2-top2
            if not .5 <= h1/h2 <= 2 or not 0 < top2-bottom1 < 3*max(h1,h2):
                continue
            pairs.append((score1+score2, (top1,bottom1), (top2,bottom2)))
    if not pairs:
        return []
    _, first, second = max(pairs)
    midpoint = (first[1]+second[0])//2
    crops = []
    for index, (top, bottom) in enumerate((first,second)):
        pad = max(4,(bottom-top)//4)
        top = max(0,top-pad,midpoint if index else 0)
        bottom = min(height,bottom+pad,midpoint if not index else height)
        crop = image.crop((0,top,width,bottom)).convert('L')
        crops.append(ImageOps.expand(crop,(10,0,10,0),'white'))
    return crops


def ordered_quad(points):
    """Order a near-horizontal convex quadrilateral as TL, TR, BR, BL."""
    points = np.asarray(points,dtype=np.float32).reshape(4,2)
    sums = points.sum(axis=1)
    differences = points[:,1]-points[:,0]
    indexes = [np.argmin(sums),np.argmin(differences),np.argmax(sums),np.argmax(differences)]
    if len(set(indexes)) != 4:
        raise ValueError('Ambiguous MRZ rectangle')
    return points[indexes]


def warp_mrz_quad(image, quad):
    """Rectify a text quadrilateral with white borders, preserving visible source pixels."""
    quad = np.asarray(quad,dtype=np.float32)
    if quad.shape != (4,2) or not np.isfinite(quad).all():
        raise ValueError('Invalid MRZ quadrilateral')
    if not cv2.isContourConvex(quad) or abs(cv2.contourArea(quad)) < 100:
        raise ValueError('Degenerate MRZ quadrilateral')
    width = int(max(np.linalg.norm(quad[1]-quad[0]),np.linalg.norm(quad[2]-quad[3])))
    height = int(max(np.linalg.norm(quad[3]-quad[0]),np.linalg.norm(quad[2]-quad[1])))
    if not (40 <= width <= 3000 and 12 <= height <= 1000):
        raise ValueError('Invalid MRZ crop size')
    destination = np.array([[0,0],[width-1,0],[width-1,height-1],[0,height-1]],np.float32)
    transform = cv2.getPerspectiveTransform(quad,destination)
    output = cv2.warpPerspective(np.asarray(image.convert('L')),transform,(width,height),
                                 flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_CONSTANT,borderValue=255)
    return Image.fromarray(output)


def perspective_mrz_candidates(image, limit=2):
    """Pair long text-line contours, allowing MRZ anywhere among other photographed IDs."""
    original = np.asarray(image.convert('L'))
    scale = min(1.,1400/original.shape[1])
    small = cv2.resize(original,None,fx=scale,fy=scale) if scale < 1 else original
    width = small.shape[1]
    # Closing joins character strokes along a row without merging the two rows.
    blackhat = cv2.morphologyEx(small,cv2.MORPH_BLACKHAT,
                                cv2.getStructuringElement(cv2.MORPH_RECT,(max(15,width//40),max(5,width//160))))
    mask = cv2.threshold(blackhat,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]
    joined = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_RECT,(max(13,width//45),3)))
    contours,_ = cv2.findContours(joined,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for contour in contours:
        center, size, angle = cv2.minAreaRect(contour)
        length, thickness = max(size), min(size)
        if thickness < 5 or not 8 < length/thickness < 85 or length < width*.28:
            continue
        if size[0] < size[1]:
            angle += 90
        angle = (angle+90)%180-90
        if abs(angle) > 25:
            continue
        # Use the original rectangle to avoid changing OpenCV's width/height convention.
        quad = ordered_quad(cv2.boxPoints(cv2.minAreaRect(contour)))
        lines.append((center,length,thickness,angle,quad))
    pairs = []
    for i, first in enumerate(lines):
        for second in lines[i+1:]:
            upper, lower = sorted((first, second), key=lambda row: row[0][1])
            c1, w1, h1, a1, q1 = upper
            c2, w2, h2, a2, q2 = lower
            if abs(a1-a2) > 10 or min(w1, w2)/max(w1, w2) < .65:
                continue
            radians = np.deg2rad((a1+a2)/2)
            normal_gap = (c2[1]-c1[1])*np.cos(radians)-(c2[0]-c1[0])*np.sin(radians)
            if not .7*max(h1, h2) < normal_gap < 4*max(h1, h2):
                continue
            if abs(c2[0]-c1[0]) > max(w1, w2)*.2:
                continue
            quad = np.array([q1[0], q1[1], q2[2], q2[3]], np.float32)
            center = quad.mean(axis=0)
            quad = (quad-center)*np.array([1.04, 1.2], np.float32)+center
            score = min(w1, w2)*(min(w1, w2)/max(w1, w2))
            pairs.append((score,quad/scale))
    selected = []
    centers = []
    for _, quad in sorted(pairs, key=lambda item: item[0], reverse=True):
        center = quad.mean(axis=0)
        if any(np.linalg.norm(center-prev) < 30/scale for prev in centers):
            continue
        try:
            crop = warp_mrz_quad(image, quad)
        except ValueError:
            continue
        if crop.width/crop.height < 3:
            continue
        selected.append(crop)
        centers.append(center)
        if len(selected) >= limit:
            break
    return selected
