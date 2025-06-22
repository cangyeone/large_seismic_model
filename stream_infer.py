import torch 
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np 


from obspy.geodetics.base import gps2dist_azimuth, kilometers2degrees, degrees2kilometers 
import obspy 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
model = torch.jit.load("pickers/multi.v5.1.jit")
model.eval()
model.to(device)

acc_time = 0
#NE 误差统计数量

stride = 20 # 每次处理20点的数据
min_prob = 0.1 # 阈值 
min_dist = 300 # 峰值最小间隔
n_phase = 4 # 震相数量
pnames = ["Pg", "Sg", "Pn", "Sn"] # 震相名称
class_level = [0, 0.1, 0.3, 0.5, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 20000.0] # 震中距估计范围
class_level = [degrees2kilometers(x) for x in class_level] # 转换为公里
etype_dict = {0:'eq', 1:'ep', 2:'ss', 3:'sp', 4:'ot', 5:'se', 6:'ve'}#地震类型

st = obspy.read("data/TASK01/T1.E001.mseed")
a1 = np.stack([x.data for x in st], axis=1)#.dtype(np.float32)
a1 = a1[None]
a1 = np.concatenate([a1, a1, a1, a1], axis=1)

B, T, C = a1.shape 
L = T//stride - 2 
batch_heaps = [[[] for i in range(n_phase)] for j in range(B)]

print("Maximum length of data:", L, T)
o_phase = []
with torch.no_grad():
    wave = torch.tensor(a1, dtype=torch.float).to(device) # 数据不做任何预处理
    wave = torch.cat([wave, wave], dim=-1)# 数据本身是三个分量，模拟成6个分量，为后续处理做准备，不影响速度
    #wave = torch.cat([wave, wave], dim=1)  # 做成更长的
    for t in range(L):#模拟实时推理
        wave_st = wave[:, t*stride:t*stride+stride*2, :]#[:, 0:6]  # 取出每次处理的20采样点三（六）分量波形数据
        o_phase_st, o_dist_class_st, o_dist_km_st, o_dist_baz_st, o_mag_st, o_etype_st = model(wave_st)
        
        o_phase_st = o_phase_st.cpu().numpy()  # 转换为numpy数组
        o_dist_class_st = o_dist_class_st.cpu().numpy()
        o_dist_km_st = o_dist_km_st.cpu().numpy() 
        o_dist_baz_st = o_dist_baz_st.cpu().numpy() 
        o_mag_st = o_mag_st.cpu().numpy() 
        o_etype_st = o_etype_st.cpu().numpy()
        o_phase.append(o_phase_st)
        for b in range(B):
            heaps = batch_heaps[b] 
            for ptype_idx, pname in enumerate(pnames):
                phase_list = heaps[ptype_idx]
                if len(phase_list)!=0:
                    pb, pidx, ptype, infos = phase_list[-1]
                else:
                    pb = 0.0
                    pidx = -1 
                    ptype = -1 
                phase_prob = o_phase_st[b, :, ptype_idx+1]
                max_idx = phase_prob.argmax()
                max_idx_stride = max_idx // stride
                max_prob = phase_prob[max_idx] 
                p_time_idx = t * stride + max_idx - 300# 延迟300采样点输出震相 
                if max_prob > min_prob:# 实时加入最大值
                    #print("SELX", p_time_idx)
                    dclass = o_dist_class_st[b, max_idx_stride, :]
                    dclass_idx = dclass.argmax() 
                    dclass_prob = dclass[dclass_idx] 
                    drange = [class_level[dclass_idx], class_level[dclass_idx+1]]
                    dist_regress = o_dist_km_st[b, max_idx_stride, :]
                    if dist_regress[0] < 300:
                        dist_km = dist_regress[0]
                        dep_km = dist_regress[1]
                    else:#大于200km时候是度
                        dist_km = degrees2kilometers(dist_regress[0]-300)  # 转换为公里
                        dep_km = dist_regress[1] 
                    baz = o_dist_baz_st[b, max_idx_stride, :]
                    mag = o_mag_st[b, max_idx_stride, 0]
                    detype = o_etype_st[b, max_idx_stride, :]
                    detype_idx = detype.argmax()
                    detype_name = etype_dict[detype_idx]
                    detype_prob = detype[detype_idx]
                    if abs(p_time_idx - pidx) < min_dist:
                        if max_prob > pb:
                            if len(phase_list) > 0:phase_list.pop()
                            phase_list.append([max_prob, p_time_idx, ptype_idx, 
                                           {
                            "phase_name":pname, 
                            "phase_prob":max_prob, 
                            "phase_time":p_time_idx,
                            "dist_range":[drange, dclass_prob],
                            "dist_km":dist_km, "dep_km":dep_km, "baz_vec":[baz[0], baz[1]], "mag":mag, 
                            "etype":[detype_name, detype_prob]
                        }])
                    else:
                        phase_list.append([max_prob, p_time_idx, ptype_idx, 
                                           {
                            "phase_name":pname, 
                            "phase_prob":max_prob, 
                            "phase_time":p_time_idx,
                            "dist_range":[drange, dclass_prob],
                            "dist_km":dist_km, "dep_km":dep_km, "baz_vec":[baz[0], baz[1]], "mag":mag, 
                            "etype":[detype_name, detype_prob]
                        }])


o_phase = np.concatenate(o_phase, axis=1)  # 合并所有批次的震相预测结果
import matplotlib.pyplot as plt 
w = a1[0, :, 2]
w = w.astype(np.float32)
w -= np.mean(w)
w /= np.max(w)
plt.plot(w, c="k", lw=1, label="Z")
plt.plot(o_phase[0, :, 1], c="r", lw=1, label="P")
plt.plot(o_phase[0, :, 2], c="b", lw=1, label="S")
phase_heaps = batch_heaps[0] 
for pidx, pname in enumerate(pnames):
    phase_list = phase_heaps[pidx]
    if len(phase_list) == 0:
        continue
    for pb, pidx, ptype, infos in phase_list:
        plt.plot([pidx, pidx], [-1, 1], c="r", lw=0.5)
        plt.text(pidx, 1.05, f"{pname} {pb:.2f}", fontsize=8, ha='center')
        print(infos)
plt.savefig("logdir/stream_infer.png", dpi=300)