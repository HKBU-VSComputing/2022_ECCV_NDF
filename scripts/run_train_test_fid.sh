echo $exp_name
which python

echo $1
echo $2

python train_net.py --cfg_file configs/zju_mocap_exp/$exp_name.yaml exp_name $exp_name resume $2
python run.py --type evaluate --cfg_file configs/zju_mocap_exp/$exp_name.yaml exp_name $exp_name resume True test_novel_pose True
python run.py --type evaluate --cfg_file configs/zju_mocap_exp/$exp_name.yaml exp_name $exp_name resume True test_novel_pose False
