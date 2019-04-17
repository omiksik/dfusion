##########################################################
# utils extracting various info from vkitti data format
#
# assumes: frame tid label truncated occluded alpha \ 
#          l t r b w3d h3d l3d x3d y3d z3d ry rx rz \
#          truncr occupr orig_label moving model color
##########################################################

from bboxes import *

# --------------------------------------------------------
def extract2Dbbox(vkitti_data_row):
# --------------------------------------------------------
	
	l = int(vkitti_data_row['l'])
	t = int(vkitti_data_row['t'])
	r = int(vkitti_data_row['r'])
	b = int(vkitti_data_row['b'])

	return BBox2DAxisAligned((l, t), (r, b))

# --------------------------------------------------------
def extract3Dbbox(vkitti_data_row):
# --------------------------------------------------------

	w3d = vkitti_data_row['w3d']
	h3d = vkitti_data_row['h3d']
	l3d = vkitti_data_row['l3d']

	x3d = vkitti_data_row['x3d']
	y3d = vkitti_data_row['y3d']
	z3d = vkitti_data_row['z3d']

	rx  = vkitti_data_row['rx']
	ry  = vkitti_data_row['ry']
	rz  = vkitti_data_row['rz']

	return BBox3D(w3d, h3d, l3d, x3d, y3d, z3d, rx, ry, rz)

# --------------------------------------------------------
def extract_tracklet_idx(vkitti_data_row):
# --------------------------------------------------------

	return vkitti_data_row['tid']