import numpy as np
import pandas as pd
import math
class Node:
    def __init__(self, isLeaf, label, threshold, gainRatio=None, majorLabel=None):
        self.label = label
        self.threshold = threshold
        self.category = []
        self.isLeaf = isLeaf
        self.children = []
        self.gainRatio = gainRatio
        self.majorLabel = majorLabel

class C45:
    def __init__(self, limitPartition=2, reuseAttribute=False):
        self.tree = None
        self.predict_label = None
        self.limitPartition = limitPartition
        self.reuseAttribute = reuseAttribute

    def fit(self, data , labels):
        """
        Xây cây.

        Đối số:
            data(DataFrame): Tập dữ liệu loại bỏ cột nhãn.
            labels(Series): Tập nhãn của data.
        """
        self.data = data
        self.data['label'] = labels
        self.columns = list(self.data.columns)
        self.dtypes = self.data.dtypes.to_list()
        self.data = self.data.values
        self.tree = self.createNode(self.data, self.columns[:-1])
        
    
    def createNode(self, data, columns):
        """
        Tạo Node hiện tại.

        Đối số:
            data(array): Mảng 2 chiều(numpy) chưa tập dữ liệu tại Node.
            columns(list): Danh sách chứa các thuộc tính của data không bao gồm thuộc tính nhãn.

        Trả về:
            Node: Node đã được tạo thành.
        """
        allSameClass = self.checkAllSameClass(data[:,-1])
        
        if allSameClass is not None:
            return Node(True, allSameClass, None)
        
        elif columns is None or len(columns) == 0:
            majorLabel = self.majorityLabel(data[:, -1])
            return Node(True, majorLabel, None)

        #  Chọn thuộc tính tốt nhất để chia
        else :
            bestAttribute, bestThreshold, bestPartitions, gainRatio = self.findBestAttribute(data, columns)

            if len(bestPartitions) == 1:
                return Node(True, self.majorityLabel(data[:, -1]), None)
            
            node = Node(False, bestAttribute, bestThreshold, gainRatio)
            node.majorLabel = self.majorityLabel(data[:, -1])
            index = columns.index(bestAttribute)
            if bestThreshold is None:
                for partition in bestPartitions:
                    node.category.append(partition[0][index])

            remainingColumns = columns[:]
            if not self.reuseAttribute:
                remainingColumns.remove(bestAttribute)

            for partition in bestPartitions:
                if not self.reuseAttribute:
                    partition = np.delete(partition, index,1)
                node.children.append(self.createNode(partition, remainingColumns))
        return node

    def findBestAttribute(self, data, columns):
        """
        Tìm thuộc tính tốt nhất để chia.

        Đối số:
            data(array): Mảng 2 chiều(numpy) chưa tập dữ liệu.
            columns(list): danh sách các thuộc tính còn lại.

        Trả về:
            bestAttribute(str): Tên thuộc tính được chọn.
            bestThreshold(list): Danh sách các giá trị ngưỡng.
            bestPartitions(list): Danh sách các đoạn dữ liệu được chia tối ưu nhất.
            gainRatio(float): Giá trị gainRatio tốt nhất cho tập dữ liệu.
        """
        splitted = []
        maxGainRatio = -1*float('inf')
        bestAttribute = -1
        threshold = None
        bestThreshold = None

        for attribute in columns:
            # labels = labels[data.index]
            index_of_attribute = columns.index(attribute)
            
            if self.isAttrDiscrete(self.dtypes[self.columns.index(attribute)]):
                unique_values = np.unique(data[:, index_of_attribute]).tolist()
                partitions = [[] for _ in range(len(unique_values))]

                for row in data:
                    partitions[unique_values.index(row[index_of_attribute])].append(row)
                
                partitions = [np.vstack(partition) for partition in partitions]

                gr = self.gainRatio(data, partitions)
                if gr >= maxGainRatio:
                    splitted = partitions
                    maxGainRatio = gr
                    bestAttribute = attribute
                    bestThreshold = None 
            
            else:     
                data = data[data[:, index_of_attribute].argsort(kind='stable')]
                threshold = []
                threshold_value=[]
                sameLabelPrevious = None
                sameLabelCurrent = data[0][-1]

                # Chọn số ngưỡng để chia
                for i in range(len(data) - 1):
                    #  Nếu 2 giá trị liền kề của thuộc tính khác nhau thì chia ngưỡng
                    if data[i][index_of_attribute] != data[i+1][index_of_attribute]:

                        # Nếu cả 2 giá trị đều có số nhãn giống nhau
                        if sameLabelPrevious is not None and sameLabelCurrent is not None and sameLabelPrevious == sameLabelCurrent:
                            threshold.pop()
                            threshold_value.pop()

                        # Thêm vào giá trị ngưỡng 
                        threshold.append(i+1)
                        threshold_value.append((data[i][index_of_attribute] + data[i+1][index_of_attribute]) / 2)
                        sameLabelPrevious = sameLabelCurrent
                        sameLabelCurrent = data[i+1][-1]

                    elif data[i][-1] != data[i+1][-1]:
                        sameLabelCurrent = None

                if sameLabelPrevious is not None and sameLabelCurrent is not None and sameLabelPrevious == sameLabelCurrent:
                    threshold.pop()
                    threshold_value.pop()

                lenThreshold = len(threshold)
                partition = 2 
                limitPartition = self.limitPartition
                if lenThreshold == 0:
                    e = self.gainRatio(data, [data])
                    if e >= maxGainRatio:
                        splitted = [data]
                        maxGainRatio = e
                        bestAttribute = attribute
                        bestThreshold = None
                else :

                    # Stack
                    for partition in range(1, self.limitPartition):
                        k = partition
                        if partition <= lenThreshold:
                            partitions = []
                            stackOfIndex = []
                            indexPre = 0
                            indexCurr = 0
                            best_threshold = []
                            flag = True
                            while flag:
                                indexPre = stackOfIndex[-1] if len(stackOfIndex) else None

                                first_threshold = threshold[indexPre] if indexPre != None else 0
                                second_threshold =  threshold[indexCurr]
                                partitions.append(data[first_threshold:second_threshold])
                                best_threshold.append(threshold_value[indexCurr])

                                stackOfIndex.append(indexCurr)
                                indexCurr += 1
                                k -=1
                                
                                if k == 0:
                                    partitions.append(data[second_threshold:])
                                    e = self.gainRatio(data, partitions)
                                    if e > maxGainRatio:
                                        splitted = partitions[:]
                                        maxGainRatio = e
                                        bestAttribute = attribute  
                                        bestThreshold = best_threshold[:]
                                    partitions.pop()
                                    while True:
                                        indexCurr = stackOfIndex.pop()
                                        partitions.pop()
                                        best_threshold.pop()
                                        k+=1
                                        if indexCurr < lenThreshold - k:
                                            indexCurr += 1
                                            break
                                        if k == partition:
                                            flag = False
                                            break

                        
                    # Trackback
                    
                    # while(partition <= limitPartition and lenThreshold >= partition-1):
                    #     indexSplits = self.fillAllTheWaySplit(lenThreshold, partition)
                    #   # Tìm được số cách chia
                    #     for indexSplit in indexSplits:
                    #         start_index = 0
                    #         partitions = []
                    #         best_threshold = []
                    #         for index in indexSplit:
                    #             partitions.append(data[start_index:threshold[index]])
                    #             best_threshold.append(threshold_value[index])
                    #             start_index = threshold[index]
                    #         partitions.append(data[start_index:])


                    #         e = self.gainRatio(data, partitions)
                    #         if e >= maxGainRatio:
                    #             splitted = partitions
                    #             maxGainRatio = e
                    #             bestAttribute = attribute  
                    #             bestThreshold = best_threshold 
                    #     partition += 1     

        return (bestAttribute, bestThreshold, splitted, maxGainRatio)
                 

    def fillAllTheWaySplit(self, lenThreshold, partition):
        """
        Tìm tất cả số cách chia.
        """
        def findNextThreshold(currIndexOfThreshold, currPartition):
            if currIndexOfThreshold == lenThreshold:
                return
            
            if currPartition == partition:
                indexSplits.append(currIndexOfThreshold)
                ans.append(indexSplits.copy())
                indexSplits.pop()
                findNextThreshold(currIndexOfThreshold + 1, currPartition)
                 
            if currPartition < partition:
                indexSplits.append(currIndexOfThreshold)
                findNextThreshold(currIndexOfThreshold + 1, currPartition + 1)
                indexSplits.pop()
                findNextThreshold(currIndexOfThreshold + 1, currPartition)
                
        ans = []
        indexSplits = []
        findNextThreshold(0, 2)
        return ans


    def isAttrDiscrete(self, dtype):
        return (dtype!= 'int64' and dtype != 'float')


    def gainRatio(self, data, partitions):
        """
        Tính gainRatio.

        Đối số:
            data(array): Tập dữ liệu cần phải tính.
            partitions(list): Danh sách các đoạn dữ liệu được chia.

        Trả về:
            float: Giá trị gainRatio tương ứng.
        """
        gain = self.gainSplit(data, partitions)
        split_info = self.splitInfo(data, partitions)
        if split_info == 0:
            gain_ratio = gain
        else:
            gain_ratio = gain / split_info
        return gain_ratio


    def gainSplit(self, data, partitions):
        """
        Tính gainSplit.
        
        Đối số:
            data(array): Tập dữ liệu cần phải tính.
            partitions(list): Danh sách các đoạn dữ liệu được chia.

        Trả về:
            float: Giá trị gainSplit tương ứng.
        """
        N = len(data)
        impurity_before = self.entropy(data)
        impurity_after = 0

        for partition in partitions:
            impurity_after += len(partition) / N * self.entropy(partition)
        total_gain = impurity_before - impurity_after
        return total_gain

    
    def entropy(self, data):
        """
        Tính entropy.

        Đối số:
            data(array): Tập dữ liệu cần phải tính.

        Trả về:
            float: Giá trị entropy tương ứng.
        """
        N = len(data)
        if N == 0:
            return 0
        _, counts = np.unique(data[:, -1], return_counts=True)
        proportions = counts / N
        proportions[proportions == 0] = 1
        entropy = 0
        for pi in proportions:
            entropy += pi * self.log(pi)
        return entropy * -1
    

    def splitInfo(self, data, partitions):
        """
        Tính splitInfo.

        Đối số:
            data(array): Tập dữ liệu cần phải tính.
            partitions(list): Danh sách các đoạn dữ liệu được chia.

        Trả về:
            float: Giá trị splitInfo tương ứng.
        """
        N = len(data)
        weights = [len(partition) / N for partition in partitions]
        split_info = 0
        for weight in weights:
            split_info += weight * self.log(weight)
        return split_info * -1


    def log(self, x):
        """
        Tính logarit cơ số x.

        Đối số:
            x(float): Giá trị cần tính.

        Trả về:
            float: Giá trị logarit cơ số 2 của x.
        """
        if x==0:
            return 0
        else:
            return math.log(x,2)

    def checkAllSameClass(self, labels):
        """
        Kiểm tra tất cả có cùng nhãn không.

        Đối số:
            labels(list): Danh sách các nhãn.

        Trả về:
            True: Tất cả nhãn giống nhau.
            False: Tồn tại nhãn khác nhau.
        """
        if labels.dtype == 'float':
            col = labels.astype(int)
            if np.isclose(col, labels).all():
                labels = col
        if len(np.unique(labels)) == 1:
            return labels[0]
        return None
    

    def majorityLabel(self, labels):
        """
        Tìm nhãn phổ biến nhất.
        
        Đối số:
            labels(list): Danh sách các nhãn.

        Trả về:
            int: Giá trị nhãn xuất hiện nhiều nhất.
        """
        if labels.dtype == 'float':
            col = labels.astype(int)
            if np.isclose(col, labels).all():
                labels = col
        labels = labels.tolist()
        return max(set(labels), key=labels.count)


    def printTree(self):
        """In ra cây."""
        self.printNode(self.tree)


    def printNode(self, node, space=""):
        """
        In ra giá trị của Node.
        
        Đối số:
            node(Node): node hiện tại.
            space(str): chuỗi khoảng trắng.
        """
        if not node.isLeaf:
            if node.threshold is None:
                # Categorical
                for index, child in enumerate(node.children):
                    if child.isLeaf:
                        print(space + "{} = {} : {}".format(node.label, node.category[index] , child.label))
                    else:
                        print(space + "{} = {}, gr = {} :".format(node.label, node.category[index] , round(node.gainRatio, 3)) )
                        self.printNode(child, space + "     ")
            else:
                for index, child in enumerate(node.children):
                    if index == 0:
                        if child.isLeaf:
                            print(space + "{} <= {} : {}".format(node.label, node.threshold[index] , child.label))
                        else:
                            print(space + "{} <= {}, gr = {} :".format(node.label, node.threshold[index] , round(node.gainRatio, 3)))
                            self.printNode(child, space + "     ")

                    elif index == len(node.children) - 1:
                        if child.isLeaf:
                            print(space + "{} > {} : {}".format(node.label, node.threshold[index - 1] , child.label))
                        else:
                            print(space + "{} > {}, gr = {} :".format(node.label, node.threshold[index - 1] , round(node.gainRatio, 3)))
                            self.printNode(child, space + "     ")

                    else:
                        if child.isLeaf:
                            print(space + "{} < {} <= {}: {}".format( node.threshold[index - 1],node.label, node.threshold[index] , child.label))
                        else:
                            print(space + "{} < {} <= {}, gr = {} :".format(node.threshold[index - 1], node.label, node.threshold[index] , round(node.gainRatio, 3)))
                            self.printNode(child, space + "     ")


    def predict(self, data):
        """
        Dự đoán nhãn của tập dữ liệu.
        
        Đối số:
            data(array): Tập dữ liệu cần dữ đoán.

        Trả về:
            list: Danh sách chứa các nhãn dự đoán.     
        """
        data = data.values
        results = []
        for row in data:
            self.predict_label = None
            self.predictRow(self.tree, row)
            results.append(self.predict_label)
        return results
    

    def predictRow(self, node, row):
        """
        Tìm node chứa giá trị dự đoán của dòng hiện tại.

        Đối số:
            node(Node): Node hiện tại.
            row(array): dòng cần dự đoán.
        """
        if not node.isLeaf:
            index_of_attribute = self.columns.index(node.label)
            if node.threshold is None:
                # Categorical
                for index, child in enumerate(node.children):
                    if row[index_of_attribute] == node.category[index]:
                        if child.isLeaf:
                            self.predict_label = child.label
                        else:
                            self.predictRow(child, row)
            else:
                # Dữ liệu liên tục
                for index, child in enumerate(node.children):
                    if index == 0:
                        if row[index_of_attribute] <= node.threshold[index]:
                            if child.isLeaf:
                                self.predict_label = child.label
                            else:
                                self.predictRow(child, row)

                    elif index == len(node.children) - 1:
                        if row[index_of_attribute] > node.threshold[index - 1]:
                            if child.isLeaf:
                                self.predict_label = child.label
                            else:
                                self.predictRow(child, row)

                    else:
                        if row[index_of_attribute] > node.threshold[index - 1] and row[index_of_attribute] <= node.threshold[index]:
                            if child.isLeaf:
                                self.predict_label = child.label
                            else:
                                self.predictRow(child, row)

    def postPruning(self, valid_data):
        valid_data = valid_data.values
        self.postPruningNode(self.tree, valid_data)
        return self.tree
    
    def postPruningNode(self, node, valid_data):
        if node.threshold is None:
            valid_partitions = self.split_data_discrete(valid_data, node.category, self.columns.index(node.label))
            for index, child in enumerate(node.children):
                if child.isLeaf:
                    continue
                elif valid_partitions[index].size != 0:
                    self.postPruningNode(child, valid_partitions[index])
        else:
            valid_partitions = self.split_data_coutinous(valid_data, node.threshold, self.columns.index(node.label))
            for index, child in enumerate(node.children):
                if child.isLeaf:
                    continue
                elif valid_partitions[index].size != 0:
                    self.postPruningNode(child, valid_partitions[index])

        for child in node.children:
            if not child.isLeaf:
                return
        accuracy_before_pruning = self.accuracy(node, valid_data)
        majorLabel = node.majorLabel
        accuracy_after_pruning = np.count_nonzero(valid_data[:, -1] == majorLabel) / len(valid_data)

        if accuracy_after_pruning >= accuracy_before_pruning:
            node.isLeaf = True
            node.label = majorLabel
            node.children = []
            node.category = []
            node.threshold = None
            node.gainRatio = None
        return

    def split_data_discrete(self, data, categories, index):
        partitions = [[] for _ in range(len(categories))]
        for row in data:
            for i in range(len(categories)):
                if row[index] == categories[i]:
                    partitions[i].append(row)
                    break
        results = []
        for partition in partitions:
            if len(partition) == 0:
                results.append(np.array([]))
            else:
                results.append(np.vstack(partition))

        return results

    def split_data_coutinous(self, data, threshold, index):
        partitions = [[] for _ in range(len(threshold) + 1)]
        for row in data:
            for i in range(len(threshold)):
                if row[index] <= threshold[i]:
                    partitions[i].append(row)
                    break
                else:
                    partitions[-1].append(row)
        results = []
        for partition in partitions:
            if len(partition) == 0:
                results.append(np.array([]))
            else:
                results.append(np.vstack(partition))

        return results
    
    def evaluate(self, data):
        return self.accuracy(self.tree, data.values)
    
    def accuracy(self, node, data):
        accuracy = 0
        for row in data:
            self.predict_label = None
            self.predictRow(node, row)
            if self.predict_label == row[-1]:
                accuracy += 1
        return accuracy / len(data)
    

from sklearn.metrics import accuracy_score
data = pd.read_csv("C:/pythonProjects/Credit_Card_Approval/train.csv")
valid_data = pd.read_csv("C:/pythonProjects/Credit_Card_Approval/validation.csv")
model = C45(limitPartition=3, reuseAttribute=False)
model.fit(data.drop(['Status'], axis=1), data['Status'])
# model.printTree()
test = pd.read_csv("C:/pythonProjects/Credit_Card_Approval/test.csv")

model.postPruning(valid_data=valid_data) 
print(model.evaluate(test))


