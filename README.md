# 红楼梦对话检索及说话人预测系统

##### 北京大学 2020年秋季学期 现代信息检索导论 小组Project
##### 小组成员： 叶其琛 赖雨亲

##### Acknowledgement:https://gitlab.com/snowhitiger/speakerextraction https://github.com/kamalkraj/BERT-SQuAD

使用说明 : 
* data 路径下存放有：
  * 红楼梦前80回文本
  * 红楼梦说话人标注数据
* 执行 `python3 PrepareSquadData.py` 获取说话人标注模型训练数据
* 执行 `python3 ExtracConversation.py` 获得红楼梦前80回对话及其上下文
* 执行 `python3 label_model_training/run_squad.py` 训练说话人标注模型，其中命令行参数参见 https://github.com/kamalkraj/BERT-SQuAD
* 执行 `python3 label_model_training/BERT.py` 对后楼梦前80回对话进行说话人标注，获得说话人预测的训练数据，并将训练数据划分为train.txt dev.txt test.txt 以及类别id映射字典：class.json（数据准备过程略去，请自行实现）
* 执行 `python3 classify_model_training/run_train.py` 获得说话人预测模型，请自行调整`bery.py`中`Config`类的各个参数（例如数据路径、模型存储路径）
* 最后在 `classify_model_training/main.py`中提供`predict()`方法用于预测说话人

BERT预训练模型获取:https://pan.baidu.com/s/1q-LbuS18Eb0M8KKThHgvbA  密码:av3o

项目前端展示：http://39.106.188.179:2333/ 