### Using labels only
 CUDA_VISIBLE_DEVICES=6,7,5,4,3,2,1,0  python step1_texmesh_train.py --datasetname fs_texmesh --name texmesh_step1 --gpu_ids 0,1,2,3,4,5,6,7 --batchSize 8