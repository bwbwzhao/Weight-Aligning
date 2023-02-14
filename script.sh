# ================================= cifar100 ===========================
# ours
python main.py \
--ds 'cifar' \
--class_num 100 \
--lr 0.1 \
--milestones 100 150 200 \
--batch_size 32 \
--epochs 250 \
--momentum 0.9 \
--weight_decay 0.0001 \
--IL_steps 20 \
--data_path '/cache/cifar/' \
--num_workers 4 \
--device 2 
# ce + ncs
python main.py \
--ds 'cifar' \
--class_num 100 \
--lr 0.1 \
--milestones 100 150 200 \
--batch_size 32 \
--epochs 250 \
--momentum 0.9 \
--weight_decay 0.0001 \
--IL_steps 5 \
--data_path '/cache/cifar/' \
--num_workers 4 \
--savedir 'cencs' --dis --device 2 
# ce + dis
python main.py \
--ds 'cifar' \
--class_num 100 \
--lr 0.1 \
--milestones 100 150 200 \
--batch_size 32 \
--epochs 250 \
--momentum 0.9 \
--weight_decay 0.0001 \
--IL_steps 5 \
--data_path '/cache/cifar/' \
--num_workers 4 \
--savedir 'cedis' --ncs --device 3
# ce
python main.py \
--ds 'cifar' \
--class_num 100 \
--lr 0.1 \
--milestones 100 150 200 \
--batch_size 32 \
--epochs 250 \
--momentum 0.9 \
--weight_decay 0.0002 \
--IL_steps 2 \
--data_path '/cache/cifar/' \
--num_workers 4 \
--savedir 'ce' --ncs --dis --device 3 
# ce + dis + wn
python main.py \
--ds 'cifar' \
--class_num 100 \
--lr 0.1 \
--milestones 100 150 200 \
--batch_size 32 \
--epochs 250 \
--momentum 0.9 \
--weight_decay 0.0002 \
--IL_steps 5 \
--data_path '/cache/cifar/' \
--num_workers 4 \
--savedir 'cediswn' --ncs --device 3
# ================================= imagenet ===========================
# ours
python main.py \
--ds 'imagenet' \
--class_num 1000 \
--lr 0.1 \
--milestones 30 60 80 90 \
--batch_size 256 \
--epochs 100 \
--momentum 0.9 \
--weight_decay 0.0001 \
--IL_steps 10 \
--data_path '/cache/imagenet/' \
--num_workers 8 \
--device 2 \
# norm=2
python main.py \
--ds 'imagenet' \
--class_num 100 \
--lr 0.1 \
--milestones 30 60 80 90 \
--batch_size 256 \
--epochs 100 \
--momentum 0.9 \
--weight_decay 0.0001 \
--IL_steps 10 \
--data_path '/cache/imagenet/' \
--num_workers 4 \
--norm 2 --savedir 'norm2' --device 0 \
# without restriction
python main.py \
--ds 'imagenet' \
--class_num 100 \
--lr 0.1 \
--milestones 30 60 80 90 \
--batch_size 256 \
--epochs 100 \
--momentum 0.9 \
--weight_decay 0.0001 \
--IL_steps 10 \
--data_path '/cache/imagenet/' \
--num_workers 4 \
--restrict \
--savedir 'worestriction' --device 1 \
# random selection
python main.py \
--ds 'imagenet' \
--class_num 100 \
--lr 0.1 \
--milestones 30 60 80 90 \
--batch_size 256 \
--epochs 100 \
--momentum 0.9 \
--weight_decay 0.0001 \
--IL_steps 10 \
--data_path '/cache/imagenet/' \
--num_workers 4 \
--random_replay \
--savedir 'random_replay' --device 2 \
# with bias
python main.py \
--ds 'imagenet' \
--class_num 100 \
--lr 0.1 \
--milestones 30 60 80 90 \
--batch_size 256 \
--epochs 100 \
--momentum 0.9 \
--weight_decay 0.0001 \
--IL_steps 10 \
--data_path '/cache/imagenet/' \
--num_workers 4 \
--withbias \
--savedir 'withbias' --device 3 \
