echo $exp_name
which python

echo $1
echo $2
echo $3

echo python -m torch.distributed.launch --nproc_per_node=2 --master_port 29545 train_net.py --cfg_file configs/zju_mocap_exp/$exp_name.yaml exp_name $exp_name resume True white_bkgd False gpus "$2,$3" distributed True
echo python run.py --type evaluate --cfg_file configs/zju_mocap_exp/$exp_name.yaml exp_name $exp_name resume True test_novel_pose True
echo python run.py --type evaluate --cfg_file configs/zju_mocap_exp/$exp_name.yaml exp_name $exp_name resume True test_novel_pose False
echo python tools/calFID.py $exp_name --start_frame 0 --end_frame $1
echo python tools/calFID.py $exp_name --start_frame $1 --end_frame 1300

python -m torch.distributed.launch --nproc_per_node=2 --master_port 29545 train_net.py --cfg_file configs/zju_mocap_exp/$exp_name.yaml exp_name $exp_name resume True white_bkgd False gpus "$2,$3" distributed True
python run.py --type evaluate --cfg_file configs/zju_mocap_exp/$exp_name.yaml exp_name $exp_name resume True test_novel_pose True
python run.py --type evaluate --cfg_file configs/zju_mocap_exp/$exp_name.yaml exp_name $exp_name resume True test_novel_pose False
python tools/calFID.py $exp_name --start_frame 0 --end_frame $1
python tools/calFID.py $exp_name --start_frame $1 --end_frame 1300
