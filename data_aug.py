# -*- coding: utf-8 -*-
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os


def opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument('--frame_sum', type=int, help='frame sum')
    parser.add_argument('--temporal_aug_num', type=int, help='temporal aug num')
    parser.add_argument('--extract_min_rate', type=float, help='extract min rate')
    parser.add_argument('--extract_max_rate', type=float, help='extract max rate')
    parser.add_argument('--min_scale_rate', type=float, help='min scale rate')
    parser.add_argument('--max_scale_rate', type=float, help='max scale rate')
    parser.add_argument('--max_principle', type=float, help='max principle')
    parser.add_argument('--special_cls', type=int, help='special cls')
    return parser.parse_args()


def get_json_files(file_pth):
    json_files = []
    for f in sorted(os.listdir(file_pth)):
        if ".json" in f:
            json_files.append(f)
    return json_files


def read_json(anno_path, show_data_stract=False):
    with open(anno_path, "r") as f:
        data = json.load(f)
        if show_data_stract:
            print("-" * 50)
            print("input file data is a dict\n\ndata.keys:")
            print(data.keys(), "\n")  # dict_keys(['meta_info', 'instance_info'])
            print("data[\"meta_info\"].keys:")
            print(data["meta_info"].keys(), "\n")
            print("data[\"instance_info\"].length:")
            print(len(data["instance_info"]), ",即总帧数")
            print("data[\"instance_info\"][i].keys:")
            print(data["instance_info"][0].keys(), "\n")
            print("data[\"instance_info\"][0][\"instances\"][0].keys:")
            print(data["instance_info"][0]["instances"][0].keys())
            print("-" * 50)


        frames = []
        keypoints = []
        keypoint_scores = []
        for f_id, frame in enumerate(data["instance_info"]):
            frame_id = frame["frame_id"]
            if len(frame["instances"]) > 1:
                # print("warning: 目标检测数据在第{}帧检测到不止一个人框，自动选择第0个人框，需要检查确认".format(f_id))
                pass
            keypoint = frame["instances"][0]["keypoints"]  # 这里考虑了可能检测到多人，但实际情况我们只考虑单人
            keypoint_score = frame["instances"][0]["keypoint_scores"]
            frames.append(frame_id)
            keypoints.append(keypoint)
            keypoint_scores.append(keypoint_score)
        frames = np.array(frames)
        keypoints = np.array(keypoints)
        keypoint_scores = np.array(keypoint_scores)
        print("frames.shape = ", frames.shape, "keypoints.shape = ", keypoints.shape, "keypoint_scores.shape = ",
              keypoint_scores.shape, "\n 可以设置show_data_stract=False关闭显示") if show_data_stract else ""

    return frames, keypoints, keypoint_scores


def norm_frame(video_frames, video_keypoints, frame_sum=200, show_detail=False):
    frames_length_list = []
    for k in video_frames.keys():
        frames_length_list.append(len(video_frames[k]))
    if show_detail:
        print("frames_length_list:", frames_length_list)
    try:
        plt.bar(video_frames.keys(), frames_length_list)
        plt.title('video frames')
        plt.xlabel('video_id')
        plt.ylabel('num of frames')
        plt.show()
        plt.savefig('输出文件/video frames.png', dpi=300)
        plt.close()

        mean = np.mean(frames_length_list)
        std = np.std(frames_length_list)
        q1 = np.percentile(frames_length_list, 25)
        q2 = np.percentile(frames_length_list, 50)
        q3 = np.percentile(frames_length_list, 75)
        median = np.median(frames_length_list)
        mode = np.argmax(np.bincount(frames_length_list))
        range_val = np.max(frames_length_list) - np.min(frames_length_list)

        indicators = ['Mean', 'Std', 'Q1', 'Q2', 'Q3', 'Median', 'Mode', 'Range']
        values = [mean, std, q1, q2, q3, median, mode, range_val]
        plt.bar(indicators, values)
        plt.title('Video Frames Data Indicators')
        plt.ylabel('Values')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
        plt.savefig('输出文件/Data Indicators.png', dpi=300)
        plt.close()
        if show_detail:
            print("mean = {}, std = {}, q1 = {}, q2 = {}, q3 = {}, median = {}, mode = {}, range_val = {}".format(mean,
                                                                                                                  std,
                                                                                                                  q1,
                                                                                                                  q2,
                                                                                                                  q3,
                                                                                                                  median,
                                                                                                                  mode,
                                                                                                                  range_val))
    except BaseException as e:
        print(e)


    frames_norm_rate = [i / frame_sum for i in frames_length_list]
    if show_detail:
        print("frames_norm_rate:", frames_norm_rate)


    for video_id, norm_rate in enumerate(frames_norm_rate):
        video_frames[video_id] = np.array([round(i * norm_rate) for i in range(frame_sum)])  # 定义采样点/插值点
        video_frames[video_id][-1] -= 1
    for video_id in video_keypoints:
        video_keypoints[video_id] = np.array(video_keypoints[video_id])[video_frames[video_id], :, :]
    return video_frames, video_keypoints


def temporal_augmentation(keypoints_seq, extract_min_rate=0.1, extract_max_rate=0.3, min_scale_rate=0.5,
                          max_scale_rate=2, max_principle=0.2, show_detail=False):

    assert len(keypoints_seq.shape) == 3 and (keypoints_seq.shape[-1] == 2 or keypoints_seq.shape[
        -1] == 3), "请确保输入数据为[num of frames, num of kp, dim of kp(2 or 3)]"
    length = keypoints_seq.shape[0]

    start = int(np.random.uniform(0, int(length * (1 - extract_min_rate))))
    gap = int(np.random.uniform(extract_min_rate * length, extract_max_rate * length))
    gap = gap if (start + gap) < length else (length - start)
    extract_rate = gap / length

    extract_seq = keypoints_seq[start: start + gap, ...]

    scale_rate = np.random.uniform(min_scale_rate, max_scale_rate)
    sample_frames = np.array([round(i / scale_rate) for i in range(int(gap * scale_rate))])
    extract_seq = extract_seq[sample_frames, ...]

    new_keypoints_seq = np.concatenate((keypoints_seq[:start, ...], extract_seq, keypoints_seq[(start + gap):, ...]),
                                       axis=0)

    seq_scale_rate = new_keypoints_seq.shape[0] / length
    frames = np.array([int(i * seq_scale_rate) for i in range(length)])
    norm_new_keypoints_seq = new_keypoints_seq[frames, ...]


    deduction_rate = extract_max_rate * max(np.abs(1 - max_scale_rate), np.abs(1 - min_scale_rate))
    deduction = extract_rate * np.abs(1 - scale_rate) * max_principle / deduction_rate
    if show_detail:
        print("采样点:\n", np.array([round(i * scale_rate + start) for i in range(gap)]))
        print("截取片段start -> end: {} -> {}, 截取比例: {}%,缩放倍数：{}".format(start, start + gap,
                                                                                 round(extract_rate * 100, 2),
                                                                                 round(scale_rate, 2)))
        print("改变片段速度后形状:", new_keypoints_seq.shape, "归一化后形状:", norm_new_keypoints_seq.shape)
        print("扣{}%分".format(round(deduction * 100, 2)))
    return norm_new_keypoints_seq, extract_rate, scale_rate, deduction


def score_uniform_sample(seq, label, gap=0.05, save_min_score=0.6, show_detail=False):
    indices_slice = []
    num_slice = []
    name_slice = []
    for i in range(int(1 / gap)):
        indices = np.where(np.logical_and(i * gap < label, label < (i + 1) * gap))
        if show_detail:
            print("{} ~ {}区间样本量为{}".format(round(i * gap, 2), round((i + 1) * gap, 2), indices[0].shape))
        indices_slice.append(indices[0])
        num_slice.append(indices[0].shape[0])
        name_slice.append("{}".format(round(i * gap, 2)))
    try:
        plt.xticks(rotation=45)
        plt.bar(name_slice, num_slice)
        plt.savefig(opt.output + "/生成样本各分数段分布.jpg", dpi=300)
        plt.close()
    except:
        pass

    if show_detail:
        print("{}~{}区间内样本量为{}, 以此数值作为采样数量, 在{}~1分之间以{}为间距均匀采样".format(
            name_slice[int(save_min_score / gap) + 1],
            name_slice[int(save_min_score / gap) + 2],
            num_slice[int(save_min_score / gap) + 1],
            save_min_score, gap))
    samples = num_slice[int(save_min_score / gap) + 1]  # 每个分数段的采样量
    sequence = []
    label_seq = []
    for i in range(int((1 - save_min_score) / gap)):
        idx = indices_slice[int(save_min_score / gap) + 1 + i][:samples]  # 进行每个分数段采样
        sequence.append(seq[idx, ...])
        label_seq.append(label[idx, ...])
        if show_detail:
            print(
                "{} ~ +{} 区间内采样后的形状为{}".format(name_slice[int(save_min_score / gap) + 1 + i], gap,
                                                         seq[idx, ...].shape))

    result_seq = np.concatenate(sequence, axis=0)
    result_label = np.concatenate(label_seq, axis=0)

    try:
        plt.bar(range(len(result_label)), result_label)
        plt.xlabel("label idx")
        plt.ylabel("score")
        plt.savefig(opt.output + "/样本均衡后分数分布.jpg", dpi=300)
        plt.close()
    except:
        pass

    indices_slice = []
    num_slice = []
    name_slice = []
    for i in range(int(1 / gap)):
        indices = np.where(np.logical_and(i * gap < result_label, result_label < (i + 1) * gap))
        if show_detail:
            print("{} ~ {}区间样本量为{}".format(round(i * gap, 2), round((i + 1) * gap, 2), indices[0].shape))
        indices_slice.append(indices[0])
        num_slice.append(indices[0].shape[0])
        name_slice.append("{}".format(round(i * gap, 2)))
    try:
        plt.xticks(rotation=45)
        plt.bar(name_slice, num_slice)
        plt.savefig(opt.output + "/均衡后生成样本各分数段分布.jpg", dpi=300)
        plt.close()
    except:
        pass
    return result_seq, result_label


def shuffle(seq, label):
    assert seq.shape[0] == label.shape[0], "时间序列样本量 != label样本量"
    idx = np.array(range(label.shape[0]))
    np.random.shuffle(idx)
    seq = seq[idx, ...]
    label = label[idx, ...]
    return seq, label



if __name__ == "__main__":
    opt = opt()
    path = opt.path
    frame_sum = opt.frame_sum

    os.makedirs(opt.output, exist_ok=True)
    json_files = get_json_files(path)

    video_frames = {}
    video_keypoints = {}
    video_keypoint_scores = {}
    for video_idx, video in enumerate(json_files):
        video_frames[video_idx], video_keypoints[video_idx], video_keypoint_scores[video_idx] = read_json(
            path + "/" + video, show_data_stract=False)


    uniform_norm_video_frames, uniform_norm_video_keypoints = norm_frame(video_frames, video_keypoints, frame_sum=200,
                                                                         show_detail=False)
    # print(type(uniform_norm_video_frames), type(uniform_norm_video_keypoints))  # <class 'dict'> <class 'dict'>
    # print(uniform_norm_video_frames.keys())  # dict_keys([0, 1, 2, 3, 4, 5, 6, 7])
    # print(uniform_norm_video_keypoints.keys())  # dict_keys([0, 1, 2, 3, 4, 5, 6, 7])
    # print(uniform_norm_video_frames[0].shape)  # (200,)
    # print(uniform_norm_video_keypoints[0].shape)  # (200, 17, 3)

    temporal_aug_num = opt.temporal_aug_num
    aug_data_keypoints_seq = []
    label = []
    for video_id in uniform_norm_video_keypoints:
        print("video_id:", video_id)
        for i in range(temporal_aug_num):
            norm_new_keypoints_seq, extract_rate, scale_rate, deduction = temporal_augmentation(
                uniform_norm_video_keypoints[video_id],
                extract_min_rate=opt.extract_min_rate,
                extract_max_rate=opt.extract_max_rate,
                min_scale_rate=opt.min_scale_rate,
                max_scale_rate=opt.max_scale_rate,
                max_principle=opt.max_principle,
                show_detail=False)
            aug_data_keypoints_seq.append(norm_new_keypoints_seq)
            label.append([1 - deduction, opt.special_cls])
    aug_data_keypoints_seq = np.array(aug_data_keypoints_seq)
    label = np.array(label)
    print("时间维度增强得到数据结果:\n", "norm_new_keypoints_seq.shape:", aug_data_keypoints_seq.shape,
          "; label.shape:",
          label.shape)

    try:
        plt.scatter(range(label.shape[0]), label, s=10)
        plt.xlabel("General Sample")
        plt.ylabel("General Score")
        plt.show()
        plt.savefig(opt.output + "/Temporal general score distribution.jpg", dpi=300)
        plt.close()
    except:
        pass


    aug_data_keypoints_seq, label = score_uniform_sample(aug_data_keypoints_seq, label, gap=0.05, show_detail=False)


    aug_data_keypoints_seq, label = shuffle(aug_data_keypoints_seq, label)

    print("时间维度增强 + 分数均匀采集 + shuffle结果:\n", "norm_new_keypoints_seq.shape:", aug_data_keypoints_seq.shape,
          "; label.shape:",
          label.shape)


    np.save(opt.output + "/aug_data_keypoints_seq,shape={}".format(aug_data_keypoints_seq.shape),
            aug_data_keypoints_seq)
    np.save(opt.output + "/aug_data_keypoints_seq_label,shape={}".format(label.shape), label)
    print(
        "seq数据已保存至: " + opt.output + "/aug_data_keypoints_seq,shape={}.npy".format(aug_data_keypoints_seq.shape))
    print("label数据已保存至: " + opt.output + "/aug_data_keypoints_seq_label,shape={}.npy".format(
        label.shape))
