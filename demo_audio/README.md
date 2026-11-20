# 界面试用录音

- `valve_id00_normal/`：MIMII 阀门 id_00 的 80 段正常录音（10 s，16 kHz，第 1 通道），用来“新建设备”。
- `test_120s_valve_id00.wav`：120 s 测试录音，由 12 段录音拼接：0–30 s 正常，30–50 s 异常，50–80 s 正常，80–90 s 异常，90–120 s 正常。
  这些测试录音没有参与建模。

试用：登录 → 新建设备（选 `valve_id00_normal` 文件夹）→ 分析录音（选 `test_120s_valve_id00.wav`）。
服务器上的结果：30–50 s 和 80–90 s 两段告警，其余时段没有误报。

来源：Hitachi, Ltd., MIMII Dataset, https://zenodo.org/records/3384388 ，CC BY-SA 4.0（本文件夹同样以 CC BY-SA 4.0 提供）。
