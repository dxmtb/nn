for lr in 0.00001;
do
    for dim in 1600;
    do
        for activation in tanh sigmoid relu; do
            for loss_type in mse softmax; do
                mkdir -p $activation-$loss_type
                CMD="python cifar_mlp.py --lr_W $lr --lr_b $lr --hidden_dim $dim --dump_prefix $activation-$loss_type-$dim-$lr --activation $activation --loss_type $loss_type"
                #echo $CMD
                echo "$CMD 1>$activation-$loss_type-$dim-$lr.log 2>&1 &"
            done    
        done
        wait
    done
done
   
