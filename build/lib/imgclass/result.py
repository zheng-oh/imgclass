import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
from itertools import cycle
import torch
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder

class Resultplt:
    def __init__(self, n_classes, label_list, score_list, matrix, labels, test_acc, preds_list,features_by_layer, target_layers=['conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'fc']):
        self.n_classes = n_classes
        self.label_list = label_list
        self.score_list = score_list
        self.matrix = matrix
        self.labels = labels
        self.test_acc = test_acc
        self.preds_list=  preds_list
        self.features_by_layer = features_by_layer
        self.target_layers = target_layers

    def conf_mat(self, plot):
        # def confusion_matrix(preds, labels, conf_matrix):
        # plt.figure(figsize=(16, 10))

        # plot.imshow(self.matrix, cmap=plt.cm.Blues)
        kk = plot.imshow(self.matrix, cmap=plt.cm.Blues)
        # 设置x轴坐标label
        plot.set_xticks([i for i in range(self.n_classes)], self.labels, rotation=45)
        # 设置y轴坐标label
        plot.set_yticks([i for i in range(self.n_classes)], self.labels)
        # 显示colorbar
        # plot.colorbar()
        plt.colorbar(kk, ax=plot)
        plot.set_xlabel("Predicted Labels")
        plot.set_ylabel("True Labels")
        # plt.title('Confusion matrix (acc='+self.summary()+')')
        plot.set_title("Confusion matrix (acc={:.2f})".format(self.test_acc))

        # 在图中标注数量/概率信息
        thresh = self.matrix.max() / 2
        for x in range(self.n_classes):
            for y in range(self.n_classes):
                # 注意这里的matrix[y, x]不是matrix[x, y]
                info = int(self.matrix[y, x])
                plot.text(
                    x,
                    y,
                    info,
                    verticalalignment="center",
                    horizontalalignment="center",
                    color="white" if info > thresh else "black",
                )
        # plot.set_tight_layout()
        # plt.savefig('./results/{}.pdf'.format(name))

    def roc(self, plt):
        score_array = np.array(self.score_list)
        # 将label转换成onehot形式
        label_tensor = torch.tensor(self.label_list)
        label_tensor = label_tensor.reshape((label_tensor.shape[0], 1))
        label_onehot = torch.zeros(label_tensor.shape[0], self.n_classes)
        label_onehot.scatter_(dim=1, index=label_tensor, value=1)
        label_onehot = np.array(label_onehot)
        fpr_dict = dict()
        tpr_dict = dict()
        roc_auc_dict = dict()
        for i in range(self.n_classes):
            fpr_dict[i], tpr_dict[i], _ = roc_curve(
                label_onehot[:, i], score_array[:, i]
            )
            roc_auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])
        # micro
        fpr_dict["micro"], tpr_dict["micro"], _ = roc_curve(
            label_onehot.ravel(), score_array.ravel()
        )
        roc_auc_dict["micro"] = auc(fpr_dict["micro"], tpr_dict["micro"])

        # macro
        # First aggregate all false positive rates
        all_fpr = np.unique(
            np.concatenate([fpr_dict[i] for i in range(self.n_classes)])
        )
        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(self.n_classes):
            mean_tpr += np.interp(all_fpr, fpr_dict[i], tpr_dict[i])
        # Finally average it and compute AUC
        mean_tpr /= self.n_classes
        fpr_dict["macro"] = all_fpr
        tpr_dict["macro"] = mean_tpr
        roc_auc_dict["macro"] = auc(fpr_dict["macro"], tpr_dict["macro"])

        # 绘制所有类别平均的roc曲线
        # plt.figure(figsize=(16, 10))
        lw = 2
        plt.plot(
            fpr_dict["micro"],
            tpr_dict["micro"],
            label="micro-average ROC curve (area = {0:0.2f})"
            "".format(roc_auc_dict["micro"]),
            color="deeppink",
            linestyle=":",
            linewidth=4,
        )

        plt.plot(
            fpr_dict["macro"],
            tpr_dict["macro"],
            label="macro-average ROC curve (area = {0:0.2f})"
            "".format(roc_auc_dict["macro"]),
            color="navy",
            linestyle=":",
            linewidth=4,
        )

        colors = cycle(["aqua", "darkorange", "cornflowerblue", "darkorchid"])
        for i, color in zip(range(self.n_classes), colors):
            plt.plot(
                fpr_dict[i],
                tpr_dict[i],
                color=color,
                lw=lw,
                label="ROC curve of class {0} (area = {1:0.2f})"
                "".format(self.labels[i], roc_auc_dict[i]),
            )
        plt.plot([0, 1], [0, 1], "k--", lw=lw)
        plt.set_xlim([0.0, 1.0])
        plt.set_ylim([0.0, 1.05])
        plt.set_xlabel("False Positive Rate")
        plt.set_ylabel("True Positive Rate")
        plt.set_title(
            "Some extension of Receiver operating characteristic to multi-class"
        )
        plt.legend(loc="lower right")
        # plt.savefig('./results/{}.pdf'.format(name))

    def max_roc(self, name):
        plt.figure(figsize=(16, 8))
        plt.tight_layout()
        ax1 = plt.subplot(122)
        self.roc(ax1)
        ax2 = plt.subplot(121)
        # ax2.margins(2, 2)           # Values >0.0 zoom out
        self.conf_mat(ax2)
        print(self.matrix)
        plt.savefig("./results/{}.pdf".format(name))


    def tsne(self, name):
        """使用 t-SNE 绘制降维后的特征图"""
        print("绘制 t-SNE 图...")

        # 检查是否捕获到特征
        if not self.features_by_layer:
            print("Error: No features captured. Please check the model and hook registration.")
            return

        # 使用 LabelEncoder 对标签进行编码（如果是字符串标签）
        label_encoder = LabelEncoder()
        all_labels_encoded = label_encoder.fit_transform(self.label_list)

        # 确保目标标签数量与特征数量匹配
        if len(all_labels_encoded) != len(self.preds_list):
            print(f"Warning: Mismatch between number of labels ({len(all_labels_encoded)}) and number of predictions ({len(self.preds_list)}).")
            return

        # 定义我们感兴趣的层

        # 创建一个包含每层 t-SNE 图的子图
        num_layers = len(self.target_layers)
        
        # 如果 target_layers 非空且长度大于 0，继续创建子图
        if num_layers == 0:
            print("Error: No layers in target_layers list.")
            return

        fig, axes = plt.subplots((num_layers + 2) // 3, 3, figsize=(5 * num_layers, 15))

        # 如果只有一层，确保 axes 是单一的 Axes 对象
        if num_layers == 1:
            axes = [axes]  # 确保 axes 始终是一个列表，即使只有一个子图

        # 对每层特征进行 t-SNE 降维并绘制
        for i, layer_name in enumerate(self.target_layers):
            if layer_name not in self.features_by_layer:
                print(f"Warning: Layer {layer_name} not found in features_by_layer.")
                continue

            features = np.concatenate(self.features_by_layer[layer_name], axis=0)  # 合并所有 batch 的特征
            # 将每层的特征展平成二维数组
            features_flattened = features.reshape(features.shape[0], -1)

            # 确保标签数量与特征数量一致
            if features_flattened.shape[0] != len(all_labels_encoded):
                print(f"Warning: The number of features ({features_flattened.shape[0]}) does not match the number of labels ({len(all_labels_encoded)}).")
                continue

            # 动态设置 perplexity，确保它小于样本数量
            n_samples = features_flattened.shape[0]
            perplexity = min(30, n_samples - 1)  # 设置 perplexity 不超过样本数量 - 1

            try:
                tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
                tsne_results = tsne.fit_transform(features_flattened)

                # Ensure axes[i] is correctly indexed as an Axes object
                scatter = axes[i // 3, i % 3].scatter(tsne_results[:, 0], tsne_results[:, 1], c=all_labels_encoded, cmap='jet', s=50)  # 调整点的大小
                axes[i // 3, i % 3].set_title(f't-SNE of {layer_name}', fontsize=20)  # 调整标题字体大小
                axes[i // 3, i % 3].set_xlabel('t-SNE component 1', fontsize=18)  # 调整标签字体大小
                axes[i // 3, i % 3].set_ylabel('t-SNE component 2', fontsize=18)  # 调整标签字体大小
                fig.colorbar(scatter, ax=axes[i // 3, i % 3])

            except ValueError as e:
                print(f"Error during t-SNE for layer {layer_name}: {e}")
                continue

        plt.tight_layout()
        plt.savefig("./results/{}.pdf".format(name))



