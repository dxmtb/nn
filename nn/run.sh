for lr in 0.1 0.01 0.001 0.0001 0.00001;
do
    python cifar_mlp.py --lr $lr 1>$lr.log 2>&1 &
done
wait
   