import torch
from fvcore.nn import FlopCountAnalysis, flop_count_table
import time


def analyze_model_flops(model, input_shape=(1, 1, 28, 28), detailed=True):
    """
    分析模型各层的FLOPS

    Parameters
    ----------
    model: 模型对象
    input_shape: 输入张量形状，默认为(1, 1, 28, 28)
    detailed: 是否显示详细层级信息

    Returns
    -------
    total_flops: 总FLOPS数
    per_layer_flops: 每层FLOPS的字典
    """
    device = next(model.parameters()).device
    input = torch.rand(input_shape).to(device)

    # 使用FlopCountAnalysis进行分析
    flops = FlopCountAnalysis(model, input)

    if detailed:
        print(flop_count_table(flops))

    return flops.total(), flops.by_module()


def analyze_prototype_flops(model, input_shape=(1, 1, 28, 28)):
    """
    特别分析原型层的FLOPS

    Parameters
    ----------
    model: 模型对象
    input_shape: 输入张量形状

    Returns
    -------
    特征提取器FLOPS，原型层FLOPS，分类层FLOPS
    """
    device = next(model.parameters()).device
    input = torch.rand(input_shape).to(device)

    # 分析特征提取
    features_flops = FlopCountAnalysis(model.features, input).total()

    # 计算feature map尺寸
    with torch.no_grad():
        features = model.features(input)

    # 分析原型层计算
    prototype_flops = 0

    # L2卷积计算
    l2_conv_flops = features.shape[1] * model.prototype_vectors.shape[0] * 2 * features.shape[2] * features.shape[3]
    prototype_flops += l2_conv_flops

    # 最小距离计算
    min_dist_flops = features.shape[0] * model.num_prototypes * features.shape[2] * features.shape[3]
    prototype_flops += min_dist_flops

    # 距离转激活
    activation_flops = model.num_prototypes
    prototype_flops += activation_flops

    # 分析最后的分类层
    with torch.no_grad():
        min_distances = torch.rand((input_shape[0], model.num_prototypes)).to(device)
        prototype_activations = model.distance_2_similarity(min_distances)
    classifier_flops = FlopCountAnalysis(model.last_layer, prototype_activations).total()

    return features_flops, prototype_flops, classifier_flops


def profile_model_execution(model, loader, num_batches=10):
    """
    分析模型执行时间和FLOPS

    Parameters
    ----------
    model: 模型对象
    loader: 数据加载器
    num_batches: 要分析的批次数量

    Returns
    -------
    执行时间和FLOPS的详细分析
    """
    model.eval()
    device = next(model.parameters()).device
    batch_count = 0

    input_shape = None

    # 获取输入形状
    for inputs, _ in loader:
        input_shape = inputs.shape
        break

    if input_shape is None:
        raise ValueError("无法确定输入形状")

    # 计算总FLOPS
    total_flops, per_layer_flops = analyze_model_flops(model, input_shape, detailed=True)
    features_flops, prototype_flops, classifier_flops = analyze_prototype_flops(model, input_shape)

    # 测量执行时间
    times = {
        "特征提取": 0,
        "原型层计算": 0,
        "分类层": 0,
        "总时间": 0
    }

    with torch.no_grad():
        for inputs, _ in loader:
            if batch_count >= num_batches:
                break

            inputs = inputs.to(device)

            # 总时间
            start = time.time()
            outputs, _ = model(inputs)
            torch.cuda.synchronize()
            times["总时间"] += time.time() - start

            # 特征提取时间
            start = time.time()
            features = model.features(inputs)
            torch.cuda.synchronize()
            times["特征提取"] += time.time() - start

            # 原型层时间
            start = time.time()
            distances = model._l2_convolution(features)
            min_distances = -torch.nn.functional.max_pool2d(
                -distances, kernel_size=(distances.size()[2], distances.size()[3])
            )
            min_distances = min_distances.view(-1, model.num_prototypes)
            prototype_activations = model.distance_2_similarity(min_distances)
            torch.cuda.synchronize()
            times["原型层计算"] += time.time() - start

            # 分类层时间
            start = time.time()
            _ = model.last_layer(prototype_activations)
            torch.cuda.synchronize()
            times["分类层"] += time.time() - start

            batch_count += 1

    # 计算平均时间
    batch_size = input_shape[0]
    samples = batch_count * batch_size

    for key in times:
        times[key] = times[key] / samples * 1000  # 转换为毫秒/样本

    # 计算FLOPS/时间比率
    flops_per_ms = {
        "特征提取": features_flops / (times["特征提取"] * batch_size) if times["特征提取"] > 0 else 0,
        "原型层计算": prototype_flops / (times["原型层计算"] * batch_size) if times["原型层计算"] > 0 else 0,
        "分类层": classifier_flops / (times["分类层"] * batch_size) if times["分类层"] > 0 else 0,
        "总计": total_flops / (times["总时间"] * batch_size) if times["总时间"] > 0 else 0
    }

    # 格式化结果
    results = {
        "批次数": batch_count,
        "样本数": samples,
        "FLOPS": {
            "特征提取": features_flops,
            "原型层计算": prototype_flops,
            "分类层": classifier_flops,
            "总计": total_flops
        },
        "时间(ms/样本)": times,
        "FLOPS/ms": flops_per_ms,
        "每部分占比": {
            "特征提取FLOPS占比": features_flops / total_flops * 100,
            "原型层FLOPS占比": prototype_flops / total_flops * 100,
            "分类层FLOPS占比": classifier_flops / total_flops * 100,
            "特征提取时间占比": times["特征提取"] / times["总时间"] * 100 if times["总时间"] > 0 else 0,
            "原型层时间占比": times["原型层计算"] / times["总时间"] * 100 if times["总时间"] > 0 else 0,
            "分类层时间占比": times["分类层"] / times["总时间"] * 100 if times["总时间"] > 0 else 0
        }
    }

    return results
