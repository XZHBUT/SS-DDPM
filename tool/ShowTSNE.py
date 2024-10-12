import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch

from Model.TestModel.LFTFormer import TransformerClassifier
from tool.DataProcess import read_OrginCWRU
from tool.MixDataset import Read_CWRU_MIX, read_Create_CWRU

def process_in_batches(model, data, batch_size=256):
    model.eval()  # 确保模型处于评估模式
    batch_outputs = []  # 用于存储所有批次的输出

    with torch.no_grad():  # 使用torch.no_grad()来减少内存消耗
        for i in range(0, len(data), batch_size):
            # 获取当前批次的数据，自动处理最后一个批次大小
            batch_data = data[i:i + batch_size].to(device)

            # 计算模型输出
            batch_output = model(batch_data)

            # 收集输出
            batch_outputs.append(batch_output)  # 如果后续处理需要在GPU上进行，可以移除.cpu()

    # 使用torch.cat将所有批次的输出拼接成一个Tensor
    return torch.cat(batch_outputs, dim=0)



def visualize_tsne(X, Y, X_labels, Y_labels, perplexity=30, learning_rate=200, n_iter=800):
    X_combined = torch.cat((X, Y), dim=0)
    labels_combined = torch.cat((X_labels, Y_labels), dim=0)

    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, n_iter=n_iter, random_state=42)
    tsne_result = tsne.fit_transform(X_combined.detach().cpu().numpy())

    tsne_X = tsne_result[:len(X)]
    tsne_Y = tsne_result[len(X):]

    plt.figure(figsize=(12, 10))

    # 调整散点大小，图例和文字大小
    scatter_real = plt.scatter(tsne_X[:, 0], tsne_X[:, 1], c=X_labels.detach().cpu().numpy(), marker='o', s=80,  # s 控制大小
                               label='Real Data', cmap='tab10', alpha=0.7)
    scatter_generated = plt.scatter(tsne_Y[:, 0], tsne_Y[:, 1], c=Y_labels.detach().cpu().numpy(), marker='^', s=80,  # s 控制大小
                                    label='Generated Data', cmap='tab10', alpha=0.7)

    # 增加图例大小和字体大小
    plt.legend(loc='best', fontsize='large')
    plt.colorbar(scatter_real)
    plt.xticks(fontsize=20)  # 增加x轴刻度的字体大小
    plt.yticks(fontsize=20)  # 增加y轴刻度的字体大小
    plt.show()


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    orign_Train_X, orign_Train_Y, orign_Valid_X, orign_Valid_Y, orign_Test_X, orign_Test_Y = read_OrginCWRU(
        filepath='../data/CWRU',
        SampleLength=2048,
        SampleNum=60,
        normal=False,
        Rate=[0.9, 0.1, 0.0]
        )

    create_Train_X, create_Train_Y, create_Valid_X, create_Valid_Y, create_Test_X, create_Test_Y = read_Create_CWRU(
        filepath='../data/CreateCWRU',
        SampleNum=60,
        Rate=[0.9, 0.1, 0.0])

    orign_Train_X = torch.tensor(orign_Train_X, dtype=torch.float32).to(device)
    orign_Train_Y = torch.tensor(orign_Train_Y, dtype=torch.int64).to(device)
    create_Train_X = torch.tensor(create_Train_X, dtype=torch.float32).to(device)
    create_Train_Y = torch.tensor(create_Train_Y, dtype=torch.int64).to(device)

    # 示例调用
    # visualize_tsne(orign_Train_X, create_Train_X, orign_Train_Y, create_Train_Y, perplexity=40, learning_rate=100,
    #                n_iter=300)


    def load_model_and_stats(path='../pth/best_model.pth'):
        state = torch.load(path)
        model = TransformerClassifier(d_model=8 , num_layers=3, num_classes=10)  # 替换为你的模型构造函数
        model.load_state_dict(state['model_state'])
        mean = state['mean']
        std = state['std']
        print(f"Model and statistics loaded from {path}")
        return model, mean, std


    # 加载整个保存的字典
    saved_state = torch.load('../pth/best_model.pth')

    # 提取模型状态字典
    model_state_dict = saved_state['model_state']

    # # 加载模型状态
    # model.load_state_dict(model_state_dict)

    # 提取并使用额外保存的统计数据
    mean = saved_state['mean']
    std = saved_state['std']
    print("Loaded model with mean:", mean, "and std:", std)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = TransformerClassifier(d_model=8 , num_layers=3, num_classes= 10)
    # model.load_state_dict(torch.load(f'../pth/best_model.pth'))
    # model.eval()

    model, mean, std = load_model_and_stats()
    model.to(device).eval()
    orign_Train_X = (orign_Train_X - mean) / std
    create_Train_X = (create_Train_X - mean) / std
    orign_Train_X = orign_Train_X.unsqueeze(1)
    create_Train_X = create_Train_X.unsqueeze(1)
    outputs_orign_Train_X = process_in_batches(model, orign_Train_X, batch_size=256)
    outputs_create_Train_X = process_in_batches(model, create_Train_X, batch_size=256)

    def compute_accuracy(output, target):
        _, predicted = torch.max(output, 1)
        correct = (predicted == target).sum().item()
        accuracy = correct / target.size(0)
        return accuracy


    a = compute_accuracy(outputs_orign_Train_X,orign_Train_Y)
    b = compute_accuracy(outputs_create_Train_X, create_Train_Y)
    print(a)
    print(b)
    visualize_tsne(outputs_orign_Train_X, outputs_create_Train_X , orign_Train_Y, create_Train_Y, perplexity=40, learning_rate=100,
                   n_iter=300)



