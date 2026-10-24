"""Geometry regressions for row pairing, perspective correction and empty images."""

import unittest
import cv2
import numpy as np
from PIL import Image, ImageDraw

from mrz_image import ordered_quad, perspective_mrz_candidates, split_mrz_lines, warp_mrz_quad


class MrzImageTests(unittest.TestCase):
    def synthetic_rows(self):
        """Create two separated text-like rows without any personal information."""
        image = Image.new('L',(1000,240),245)
        draw = ImageDraw.Draw(image)
        for top in (70,130):
            for column in range(44):
                x=50+column*20
                draw.rectangle((x,top,x+12,top+20),fill=15)
        return image

    def test_blank_image_has_no_row_or_geometric_candidates(self):
        image=Image.new('L',(1000,240),'white')
        self.assertEqual(split_mrz_lines(image),[])
        self.assertEqual(perspective_mrz_candidates(image),[])

    def test_two_separated_rows_become_two_single_line_crops(self):
        rows=split_mrz_lines(self.synthetic_rows())
        self.assertEqual(len(rows),2)
        for row in rows:
            self.assertLess(row.height,90)
            self.assertLess(np.asarray(row).min(),30)

    def test_single_row_is_not_invented_into_two_lines(self):
        image=self.synthetic_rows()
        ImageDraw.Draw(image).rectangle((0,110,999,239),fill=245)
        self.assertEqual(split_mrz_lines(image),[])

    def test_long_rows_are_rectified_as_a_pair(self):
        candidates=perspective_mrz_candidates(self.synthetic_rows())
        self.assertGreaterEqual(len(candidates),1)
        self.assertLessEqual(len(candidates),2)
        self.assertEqual(len(split_mrz_lines(candidates[0])),2)

    def test_passport_in_upper_part_is_not_discarded(self):
        image=Image.new('L',(1000,700),245)
        image.paste(self.synthetic_rows(),(0,0))
        self.assertTrue(perspective_mrz_candidates(image))

    def test_thin_rotated_rectangle_keeps_long_edge_horizontal(self):
        points=cv2.boxPoints(((150.,150.),(240.,12.),20.))
        ordered=ordered_quad(points)
        long_edge=np.linalg.norm(ordered[1]-ordered[0])
        short_edge=np.linalg.norm(ordered[3]-ordered[0])
        self.assertGreater(long_edge,short_edge*10)

    def test_identity_warp_preserves_corner_regions(self):
        image=Image.new('L',(200,100),255)
        draw=ImageDraw.Draw(image)
        draw.rectangle((0,0,99,49),fill=20)
        draw.rectangle((100,0,199,49),fill=80)
        draw.rectangle((100,50,199,99),fill=140)
        draw.rectangle((0,50,99,99),fill=200)
        output=warp_mrz_quad(image,[[0,0],[199,0],[199,99],[0,99]])
        self.assertLess(output.getpixel((10,10)),30)
        self.assertAlmostEqual(output.getpixel((180,10)),80,delta=5)
        self.assertAlmostEqual(output.getpixel((180,80)),140,delta=5)
        self.assertAlmostEqual(output.getpixel((10,80)),200,delta=5)

    def test_invalid_quadrilaterals_are_rejected(self):
        image=self.synthetic_rows()
        for quad in ([[0,0]]*4,[[0,0],[10,10],[0,10],[10,0]],[[0,0],[1,0],[1,1],[0,1]]):
            with self.subTest(quad=quad),self.assertRaises(ValueError):
                warp_mrz_quad(image,quad)


if __name__=='__main__':
    unittest.main()
