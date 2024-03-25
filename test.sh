CUDA_VISIBLE_DEVICES=0 python3 test.py --dataset dtu --batch_size 1 \
                                       --testpath ./dtu \
                                       --testlist ./lists/dtu/test.txt \
                                       --resume ./pretrained_weights/MVSFormer/MVSFormer/best.pth \
                                       --outdir ./output \
                                       --fusibile_exe_path ../fusibile/fusibile \
                                       --interval_scale 1.06 --num_view 5 \
                                       --numdepth 192 --max_h 1152 --max_w 1536 --filter_method gipuma \
                                       --disp_threshold 0.1 --num_consistent 2 \
                                       --prob_threshold 0.5,0.5,0.5,0.5 \
                                       --combine_conf --tmps 5.0,5.0,5.0,1.0
                                       
                                       
CUDA_VISIBLE_DEVICES=0 python3 demo_ros.py --dataset dtu --batch_size 1 \
                                       --testpath ./dtu \
                                       --testlist ./lists/dtu/test.txt \
                                       --resume ./pretrained_weights/MVSFormer/MVSFormer/best.pth \
                                       --outdir ./output \
                                       --fusibile_exe_path ../fusibile/fusibile \
                                       --interval_scale 1.06 --num_view 5 \
                                       --numdepth 192 --max_h 1152 --max_w 1536 --filter_method gipuma \
                                       --disp_threshold 0.1 --num_consistent 2 \
                                       --prob_threshold 0.5,0.5,0.5,0.5 \
                                       --combine_conf --tmps 5.0,5.0,5.0,1.0
                                       
                                       
                                       
                                       
CUDA_VISIBLE_DEVICES=0 python3 demo.py --dataset dtu --batch_size 1 \
                                       --testpath ./dtu \
                                       --testlist ./lists/dtu/test.txt \
                                       --resume ./pretrained_weights/MVSFormer/MVSFormer/best.pth \
                                       --outdir ./output \
                                       --fusibile_exe_path ../fusibile/fusibile \
                                       --interval_scale 1.06 --num_view 5 \
                                       --numdepth 192 --max_h 1152 --max_w 1536 --filter_method gipuma \
                                       --disp_threshold 0.1 --num_consistent 2 \
                                       --prob_threshold 0.5,0.5,0.5,0.5 \
                                       --combine_conf --tmps 5.0,5.0,5.0,1.0
