#!/usr/local/bin/python3

# Define the Tree and Node
class Tree:
	def __init__(self, cargo, left=None, right=None):
		self.cargo = cargo
		self.left = left
		self.right = right

	def __str__(self):
		return str(self.cargo)

# Print the Tree formula
def print_tree(tree):
	if tree is None: return
	print_tree(tree.left)
	print(tree.cargo['Value'], end="")
	print_tree(tree.right)

# Print the Tree structure
# (Note: The most left node is the root, the most right nodes are the leaves.
# Downward is left, upward is right)
def print_tree_indented(tree, level=0):
	if tree is None: return
	print_tree_indented(tree.right, level+1)
	print("---|" * level + str(tree.cargo['Value']))
	print_tree_indented(tree.left, level+1)

# calculate the number of layers
def calculate_layers(tree, current=0):
	if tree is None: return current
	left  = calculate_layers(tree.left, current+1)
	right = calculate_layers(tree.right, current+1)
	return max(left, right)


def EncodeSuccint(root, struc, data):
	# If root is None , put 0 in structure array and return
	if root is None:
		struc.append(0)
		return

	# Else place 1 in structure array, key in 'data' array
	# and recur for left and right children
	if root.cargo['Value'] == 'and':
		struc.append(1)
	elif root.cargo['Value'] == 'or':
		struc.append(2)
	elif root.cargo['Value'] == 'alw':
		struc.append(3)
	elif root.cargo['Value'] == 'ev':
		struc.append(4)
	elif root.cargo['Value'][1] in ['>', '>=', '<', '<=']:
		struc.append(5)
	else:
		print('Invalid formulas ')

	data.append(root.cargo)
	EncodeSuccint(root.left, struc, data)
	EncodeSuccint(root.right, struc, data)


# Constructs tree from 'struc' and 'data'
def DecodeSuccinct(struc, data):
	if (len(struc) <= 0):
		return None

	# Remove one item from structure list
	b = struc[0]
	struc.pop(0)

	# If removed bit is 1
	if b >= 1:
		cargo = data[0]
		data.pop(0)

		# Create a tree node with removed data
		root = Tree(cargo)

		# And recur to create left and right subtrees
		root.left = DecodeSuccinct(struc, data)
		root.right = DecodeSuccinct(struc, data)
		return root

	return None




