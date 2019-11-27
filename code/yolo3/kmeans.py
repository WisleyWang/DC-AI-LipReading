import numpy as np


class YOLO_Kmeans:

    def __init__(self, cluster_number, filename):
        self.cluster_number = cluster_number
        self.filename = filename

    def iou(self, boxes, clusters):  # 1 box -> k clusters_shape=(k,2) ,boxes=(-1,2)
        n = boxes.shape[0]
        k = self.cluster_number

        box_area = boxes[:, 0] * boxes[:, 1]  #盒子的面积
        box_area = box_area.repeat(k)        #每个盒子重复K次，因为有K个中心
        box_area = np.reshape(box_area, (n, k))  #shape=(-1,k) 每行都是重复的当前盒子面积

        cluster_area = clusters[:, 0] * clusters[:, 1]   #中心盒子的距离
        cluster_area = np.tile(cluster_area, [1, n])    #shape=（k,n） K个中心n个盒子
        cluster_area = np.reshape(cluster_area, (n, k))  #shape=（n,K） K个中心n个盒子面积

        box_w_matrix = np.reshape(boxes[:, 0].repeat(k), (n, k)) #每行都是重复的当前盒子的宽
        cluster_w_matrix = np.reshape(np.tile(clusters[:, 0], (1, n)), (n, k)) ##每行都是一样的K个中心的宽
        min_w_matrix = np.minimum(cluster_w_matrix, box_w_matrix)  #找两个框的最小宽长

        box_h_matrix = np.reshape(boxes[:, 1].repeat(k), (n, k))  #同上找出最小长
        cluster_h_matrix = np.reshape(np.tile(clusters[:, 1], (1, n)), (n, k))
        min_h_matrix = np.minimum(cluster_h_matrix, box_h_matrix)
        inter_area = np.multiply(min_w_matrix, min_h_matrix)  #最小重复面积

        result = inter_area / (box_area + cluster_area - inter_area)
        return result  #shape=(n,k)

    def avg_iou(self, boxes, clusters):
        accuracy = np.mean([np.max(self.iou(boxes, clusters), axis=1)])
        return accuracy  #shape=(1)  所有所有n个盒子的最近中心的平均

    def kmeans(self, boxes, k, dist=np.median):     #shape=（-1，2） np.median求中位数，K为聚类数
        box_number = boxes.shape[0]    #所有盒子的数量
        distances = np.empty((box_number, k))
        last_nearest = np.zeros((box_number,))
        np.random.seed()
        clusters = boxes[np.random.choice(
            box_number, k, replace=False)]  # init k clusters   #无重复抽样，K个样本均不同。kmean第一步，随机选k个点
        while True:

            distances = 1 - self.iou(boxes, clusters)   #计算所有盒子到K个中心点的距离  iou越大 距离越小


            current_nearest = np.argmin(distances, axis=1)   #找出每个盒子其最近的一个距离  得到（-1，1） 这里的1是最近点的那个索引
            if (last_nearest == current_nearest).all():  #如果每个盒子的最近点没有改变，则结束计算
                break  # clusters won't change
            for cluster in range(k):          #
                clusters[cluster] = dist(  # update clusters
                    boxes[current_nearest == cluster], axis=0)  #先找出最近的点是该点的所有盒子，在计算这些盒子的中位数，得到该类盒子的w和h

            last_nearest = current_nearest

        return clusters

    def result2txt(self, data):
        f = open("yolo_anchors.txt", 'w')
        row = np.shape(data)[0]
        for i in range(row):
            if i == 0:
                x_y = "%d,%d" % (data[i][0], data[i][1])
            else:
                x_y = ", %d,%d" % (data[i][0], data[i][1])
            f.write(x_y)
        f.close()

    def txt2boxes(self):
        with open(self.filename, 'r') as f:
            infos = f.readlines()
        dataSet = []

        length = len(infos)
        for i in range(1, length):
            width = int(infos[i].split(",")[3]) - \
                int(infos[i].split(",")[1])
            height = int(infos[i].split(",")[4]) - \
                int(infos[i].split(",")[2])
            dataSet.append([width, height])   #将所有文件的所有box都装在列表里
        result = np.array(dataSet)    #shape=（-1，2）
        return result

    def txt2clusters(self):
        all_boxes = self.txt2boxes()   #shape=（-1，2）
        result = self.kmeans(all_boxes, k=self.cluster_number)  #聚类个数就是anchor的个数
        result = result[np.lexsort(result.T[0, None])]
        self.result2txt(result)
        print("K anchors:\n {}".format(result))
        print("Accuracy: {:.2f}%".format(
            self.avg_iou(all_boxes, result) * 100))
        return self.avg_iou(all_boxes, result) * 100


if __name__ == "__main__":
    cluster_number = 9  # 聚类数
    filename = "./manul_label.txt"
    kmeans = YOLO_Kmeans(cluster_number, filename)
    acc=0
    while (acc<85.5): # acc大于85.5就停止
        acc=kmeans.txt2clusters()

