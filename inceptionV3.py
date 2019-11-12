'''
import tensorflow as tf
import os
import numpy as np
import re
from PIL import Image
import matplotlib.pyplot as plt


# 创建节点类
class NodeLookup(object):
    def __init__(self):
        label_lookup_path = 'inception-2015-12-05/imagenet_2012_challenge_label_map_proto.pbtxt'
        uid_lookup_path = 'inception-2015-12-05/imagenet_synset_to_human_label_map.txt'
        self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

    def load(self, label_lookup_path, uid_lookup_path):
        # 载入分类名称文件
        proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path, 'r').readlines()
        uid_to_human = {}  # 初始化字典结构
        for line in proto_as_ascii_lines:
            # 去除换行符
            line = line.strip('\n')
            # 按'\t'进行分割
            parse_items = line.split('\t')
            # 分类编号：类似于n00004475
            uid = parse_items[0]
            # 标签
            human_string = parse_items[1]
            # 加入字典结构
            uid_to_human[uid] = human_string

        # 载入分类编号文件
        proto_as_ascii = tf.gfile.GFile(label_lookup_path, 'r').readlines()
        nodeid_to_uid = {}  # 初始化字典结构
        for line in proto_as_ascii:
            # 去除换行符
            line = line.strip('\n')
            # 寻找target_class
            if line.startswith('  target_class:'):
                target_class = int(line.split(': ')[1])
            # 寻找target_class_string
            if line.startswith('  target_class_string:'):
                target_class_string = line.split(': ')[1]
                # 加入字典结构
                nodeid_to_uid[target_class] = target_class_string[1:-1]

        # 进行字典结构关联
        nodeid_to_name = {}
        for key, val in nodeid_to_uid.items():
            # 获取分类名称
            name = uid_to_human[val]
            # 建立分类编号1-1000到分类名称的映射关系
            nodeid_to_name[key] = name

        return nodeid_to_name

    # #根据传入分类编号1-1000返回分类名称
    def id_to_string(self, nodeid):
        if nodeid not in self.node_lookup:
            return ''
        return self.node_lookup[nodeid]


# 创建一个图来存放google训练好后的模型
with tf.gfile.FastGFile('inception-2015-12-05/classify_image_graph_def.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

# 创建会话
with tf.Session() as sess:
    # inception-v3最后一层的tensor
    softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
    # 遍历整个待测试目录
    for root, dirs, files in os.walk("inception-2015-12-05/", topdown=False):
        # 遍历该目录下所有图片
        for file in files:
            # 载入图片
            image_data = tf.gfile.GFile(os.path.join(root, file), 'rb').read()
            predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
            print('zz:', predictions.shape)  # (1, 1008)
            predictions = np.squeeze(predictions)  # 把结果转化为1维数据
            print('xx:', predictions.shape)  # (1008,)

            # 打印图片路径及名称
            image_path = os.path.join(root, file)
            print(image_path)
            # 显示图片
            img = Image.open(image_path)
            print('img:', type(img))
            plt.imshow(img)
            plt.axis('off')
            plt.show()

            # 排序：从大到小前5个排序
            top_k = predictions.argsort()[-5:][::-1]
            print('top_k:', top_k)  # top_k: [274 268 273 382 563]
            node_lookup = NodeLookup()
            for nodeid in top_k:
                # 分类名称
                human_string = node_lookup.id_to_string(nodeid)
                # 获取分类的置信度
                score = predictions[nodeid]
                print('%s (score = %.5f)' % (human_string, score))
0-            print('\n')
'''
'''
import tensorflow as tf
import numpy as np

uid_to_human = {}
for line in tf.gfile.GFile('output_labels.txt').readlines():
	items = line.strip().split('\t')
	uid_to_human[items[0]] = items[1]

node_id_to_uid = {}
for line in tf.gfile.GFile('inception-2015-12-05/imagenet_2012_challenge_label_map_proto.pbtxt').readlines():
	if line.startswith('  target_class:'):
		target_class = int(line.split(': ')[1])
	if line.startswith('  target_class_string:'):
		target_class_string = line.split(': ')[1].strip('\n').strip('\"')
		node_id_to_uid[target_class] = target_class_string

node_id_to_name = {}
for key, value in node_id_to_uid.items():
	node_id_to_name[key] = uid_to_human[value]

def create_graph():
	with tf.gfile.FastGFile('inception-2015-12-05/classify_image_graph_def.pb', 'rb') as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())
		_ = tf.import_graph_def(graph_def, name='')

def classify_image(image, top_k=1):
	image_data = tf.gfile.FastGFile(image, 'rb').read()

	create_graph()

	with tf.Session() as sess:
		# 'softmax:0': A tensor containing the normalized prediction across 1000 labels
		# 'pool_3:0': A tensor containing the next-to-last layer containing 2048 float description of the image
		# 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG encoding of the image
		softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
		predictions = sess.run(softmax_tensor, feed_dict={'DecodeJpeg/contents:0': image_data})
		predictions = np.squeeze(predictions)

		top_k = predictions.argsort()[-top_k:]
		for node_id in top_k:
			human_string = node_id_to_name[node_id]
			score = predictions[node_id]
			print('%s (score = %.5f)' % (human_string, score))

classify_image('renametrain/0_0.jpg')
'''
import tensorflow as tf
import os
import numpy as np
import re
from PIL import Image
import matplotlib.pyplot as plt

lines = tf.gfile.GFile('output_labels.txt').readlines()
uid_to_human = {}

for uid, line in enumerate(lines):
	line = line.strip('\n')
	uid_to_human[uid] = line

def id_to_string(node_id):
	if node_id not in uid_to_human:
		return ''
	return uid_to_human[node_id]

with tf.gfile.GFile('output_graph.pb', 'rb') as f:
	graph_def = tf.GraphDef()
	graph_def.ParseFromString(f.read())
	tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
	softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

	#遍历目录
	for root, dirs, files in os.walk('renametest/'):
		for file in files:
			image_data = tf.gfile.GFile(os.path.join(root, file), 'rb').read()
			predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
			predictions = np.squeeze(predictions)
			image_path = os.path.join(root, file)
			print(image_path)


			top_k = predictions.argsort()[::-1]
			for node_id in top_k:
				human_string = id_to_string(node_id)

				score = predictions[node_id]
				print('%s (score = %.5f)' % (human_string, score))
			print()

			img = Image.open(image_path)
			plt.imshow(img)
			plt.axis('off')
			plt.show()