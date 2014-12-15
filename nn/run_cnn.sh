for activation in tanh sigmoid relu; do
    for loss_type in mse softmax; do
        echo "python cifar_cnn.py --dump_prefix cnn/new-$activation-$loss_type --activation $activation --loss_type $loss_type 1>cnn/new-$activation-$loss_type.log --load_path cnn/$activation-$loss_type---epoch-99.model 2>&1"
    done    
done
wait
