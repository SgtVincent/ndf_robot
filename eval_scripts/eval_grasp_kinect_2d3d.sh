source /home/junting/sslgrasp/ndf_robot/ndf_env.sh

demos=('688eacb56a16702f9a6e43b878d5b335' '912a49eeda6cd8c5e3c462ac6e0f506f' \
    '6b8b2cb01c376064c8724d5673a063a6' '2bbd2b37776088354e23e9314af9ae57' \
    '32074e5642bad0e12c16495e79df12c1' 'b95559b4f2146b6a823177feb9bf5114' \
    '75bf30ec68b8477e7099d25c8e75cf58' '5d6b5d1b135b8cf8b7886d94372e3c76' \
    'b1271f402910cf05cfdfe3f21f42a111' 'd5dd0b4d16d2b6808bda158eedb63a62' \
    '7778c06ab2af1121b4bfcf9b3e6ed915' 'dc0926ce09d6ce78eb8e919b102c6c08' \
    '7d41c6018862dc419d231837a704886d' '32bb26b4d2cd5ee0b3b14ef623ad866a' \
    '204c2b0a5e0a0ac3efd4121045846f28' '991d9df5bf2a33a1c9292f26f73f6538' \
    'c46bfae78beaa4a7988abef1fd117e7' 'af3dda1cfe61d0fc9403b0d0536a04af' \
    '4301fe10764677dcdf0266d76aa42ba' 'b2498accc1c3fe732db3066d0100ee4')

run_exp(){
    python src/ndf_robot/eval/evaluate_ndf_grasp_only.py \
                    --demo_exp grasp_side_place_shelf_start_upright_all_methods_multi_instance \
                    --object_class bottle \
                    --config eval_bottle_gen \
                    --exp eval_grasp_bottle \
                    --opt_iterations 500 \
                    --model_path multi_category_weights \
                    --only_test_ids \
                    --num_cams 1 \
                    --depth_noise kinect \
                    --demo_filter $1 \
                    --modality 2d3d \
                    --pos_fusion 2d
}

for demo in ${demos[@]}; do
    echo "\n\n----------------start running experiment with demo $demo--------------\n\n"
    run_exp $demo
done