# Template for training an ensemble of 4 MACE models (one per GPU)

# Step 1 : force_weight = 100, energy_weight = 1, lr = 0.01 

for i in {00..03}; do 

    printf -v model 'model-%02d' "$i"
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    echo "~~~~~~~~~~~~ training ${model} ~~~~~~~~~~~~~~~~~~~~~"
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

    printf -v model 'model-%02d' "$i"
    seed=$((i+1))
    name=${PWD##*/}

    args=(
    # SETUP
        --name="${model}"                      
        --seed="${seed}"
        #--log_dir='logs'
        #--model_dir='.'
        #--checkpoints_dir='checkpoints'
        #--results_dir='results'
        #--downloads_dir='downloads'
        --device='cuda'
        --default_dtype='float32'
        #--log_level='INFO'
        --error_table='PerAtomRMSE' 

    # MODEL
        --model='MACE'                
        --r_max=6.0
        #--radial_type='bessel'
        #--num_radial_basis=8
        #--num_cutoff_basis=5
        #--interaction='RealAgnosticResidualInteractionBlock'
        #--interaction_first='RealAgnosticResidualInteractionBlock'
        #--max_ell=3
        #--correlation=3
        #--num_interactions=2
        #--MLP_irreps='16x0e'
        #--radial_MLP='[64,64,64]'
        #--hidden_irreps='128x0e + 128x1o'
        --num_channels=256
        --max_L=0
        #--gate='silu'
        #--scaling='rms_forces_scaling'
        #--avg_num_neighbors=1
        #--compute_avg_num_neighbors # default: True
        #--compute_stress # default: False
        #--compute_forces # default: True

    # DATASET
        --train_file="data/train_$i.xyz"    
        --valid_file="data/valid_$i.xyz" 
        --E0s='average' #None

    # LOSS
        #--energy_key='energy'                    
        #--forces_key='forces'
        #--virials_key='virials'
        #--stress_key='stress'
        #--dipole_key='dipole'
        #--charges_key='charges'
        --loss='weighted'
        --forces_weight=100.0
        #--swa_forces_weight=100.0
        --energy_weight=1.0
        #--swa_energy_weight=1000.0
        #--virials_weight=1.0
        #--swa_virials_weight=10.0
        #--stress_weight=1.0
        #--swa_stress_weight=10.0
        #--dipole_weight=1.0
        #--charges_weight=1.0
        #--swa_dipole_weight=1.0
        #--swa_charges_weight=1.0
        #--config_type_weights='{"Default":1.0}'
        #--huber_delta=0.01
        #--atom_group_weights=None
        #--element_group_weights=None

    # OPTIMIZATION
        #--optimizer='adam'                       
        --batch_size=8
        --valid_batch_size=10
        --lr=0.01
        #--swa_lr=0.001
        #--weight_decay=5e-07
        #--amsgrad #default: True
        #--scheduler='ReduceLROnPlateau'
        #--lr_factor=0.8
        #--scheduler_patience=100
        #--lr_scheduler_gamma=0.9993
        #--swa #default: False
        #--start_swa=800
        #--ema #default: False
        #--ema_decay=0.99
        --max_num_epochs=800
        #--patience=50
        --eval_interval=1
        #--keep_checkpoints #default: False
        --restart_latest #default: False
        --save_cpu #default: False
        #--clip_grad=10.0

    # WANDB
        # --wandb #default: False                            
        # --wandb_project='FeCo'
        # --wandb_name="$name"
        # --wandb_log_hypers="['model','r_max','num_channels','max_L','correlation','num_interactions','lr','swa_lr','weight_decay','batch_size','max_num_epochs','start_swa','energy_weight','forces_weight','charges_weight','element_group_weights' ]"
    )
    
    CUDA_VISIBLE_DEVICES=$i CUDA_VISIBLE_DEVICES=$i mace_run_train "${args[@]}" &

done
wait

