import unittest
import numpy as np
from region_affinity import merge_regions,region_pairs


class RegionAffinityTests(unittest.TestCase):
    def test_no_link_keeps_partition_and_pallet(self):
        grid=np.array([[1,2,2,3],[1,2,4,4]])
        out=merge_regions(grid,np.array([2,3,4]),np.array([[0,1],[0,2],[1,2]]),[.9,.9,.9],1.)
        np.testing.assert_array_equal(grid,out)

    def test_complete_link_prevents_weak_transitive_merge(self):
        grid=np.array([[1,2,3,4]])
        out=merge_regions(grid,np.array([2,3,4]),np.array([[0,1],[0,2],[1,2]]),[.99,.1,.95],.9)
        self.assertEqual(out[0,1],out[0,2])
        self.assertNotEqual(out[0,2],out[0,3])
        self.assertEqual(out[0,0],1)

    def test_degenerate_region_features_are_finite(self):
        g=np.array([[2,3],[1,1]])
        r=dict(valid=np.ones((2,2),bool),means=np.zeros((2,2,7)),lo=np.zeros(2),cell=4.)
        ids,pairs,x=region_pairs(r,g)
        self.assertEqual(x.shape[0],1)
        self.assertTrue(np.isfinite(x).all())


if __name__=='__main__':unittest.main()
