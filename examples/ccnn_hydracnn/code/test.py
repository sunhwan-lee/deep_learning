#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

"""
Test script. This code performs a test over with a pre trained model over the
specified dataset.
"""

#===========================================================================
# Dependency loading
#===========================================================================
# File storage
import h5py
import scipy.io as sio
import skimage.io
from matplotlib import pyplot as plt
from matplotlib import cm

# System
import signal
import sys, getopt, os
signal.signal(signal.SIGINT, signal.SIG_DFL)
import time

# Vision and maths
import numpy as np
import utils as utl
from gen_features import genDensity, genPDensity, loadImage, extractEscales

# tensorflow
import tensorflow as tf

#===========================================================================
# Code 
#===========================================================================
def load_model(tfdata, tfclass, tfmodule):

  # append tensorflow convertor module
  sys.path.append(os.path.abspath(tfmodule))
  # append path to tensorflow class file
  sys.path.append(os.path.abspath(tfclass[:tfclass.rfind("/")+1]))

  tfclass_name = tfclass[tfclass.rfind("/")+1:]    
  
  if "trancos" in tfclass_name:
    if "ccnn" in tfclass_name:
  
      from trancos_ccnn import TRANCOS_CCNN
  
      images = tf.placeholder(tf.float32, shape=(None, 72, 72, 3))
      net = TRANCOS_CCNN({"data_s0": images})

      weights = np.load(tfdata, encoding="latin1")
      weights.item()["conv6"]["biases"] = np.array([weights.item()["conv6"]["biases"]])

  return net, weights
        
def gameRec(test, gt, cur_lvl, tar_lvl):
    '''
    @brief: Compute the game metric error. Recursive function.
    @param test: test density.
    @param gt: ground truth density.
    @param cur_lvl: current game level.
    @param tar_lvl: target game level.
    @return game: return game metric.
    '''
        
    # get sizes
    dim = test.shape
    
    assert dim == gt.shape
    
    if cur_lvl == tar_lvl:
        return np.abs( np.sum( test ) - np.sum( gt ) )
    else:

        # Creating the four slices
        y_half = int( dim[0]/2 )
        x_half = int( dim[1]/2 )
        
        dens_slice = []
        dens_slice.append( test[ 0:y_half, 0:x_half ] )
        dens_slice.append( test[ 0:y_half, x_half:dim[1] ] )
        dens_slice.append( test[ y_half:dim[0], 0:x_half] )
        dens_slice.append( test[ y_half:dim[0], x_half:dim[1] ] )

        gt_slice = []
        gt_slice.append( gt[ 0:y_half, 0:x_half ] )
        gt_slice.append( gt[ 0:y_half, x_half:dim[1] ] )
        gt_slice.append( gt[ y_half:dim[0], 0:x_half] )
        gt_slice.append( gt[ y_half:dim[0], x_half:dim[1] ] )

        res = np.zeros(4)
        for a in range(4):
            res[a] = gameRec(dens_slice[a], gt_slice[a], cur_lvl + 1, tar_lvl)
    
        return np.sum(res);      

'''
    @brief: Compute the game metric error.
    @param test: test density.
    @param gt: ground truth density.
    @param lvl: game level. lvl = 0 -> mean absolute error.
    @return game: return game metric.
'''
def gameMetric(test, gt, lvl):
    return gameRec(test, gt, 0, lvl)        

#===========================================================================
# Some helpers functions
#===========================================================================
def initTestFromCfg(cfg_file):
  '''
  @brief: initialize all parameter from the cfg file. 
  '''
    
  # Load cfg parameter from yaml file
  cfg = utl.cfgFromFile(cfg_file)
  
  # Fist load the dataset name
  dataset = cfg.DATASET
  
  # Set default values
  use_mask = cfg[dataset].USE_MASK
  use_perspective = cfg[dataset].USE_PERSPECTIVE
  
  # Mask pattern ending
  mask_file = cfg[dataset].MASK_FILE
      
  # Img patterns ending
  dot_ending = cfg[dataset].DOT_ENDING
  
  # Test vars
  test_names_file = cfg[dataset].TEST_LIST
  
  # Im folder
  im_folder = cfg[dataset].IM_FOLDER
  
  # Results output foder
  results_file = cfg[dataset].RESULTS_OUTPUT

  # Resize image
  resize_im = cfg[dataset].RESIZE

  # Patch parameters
  pw = cfg[dataset].PW # Patch with 
  pw_norm = cfg[dataset].CNN_PW_IN  # Patch width
  sigmadots = cfg[dataset].SIG # Densities sigma
  n_scales = cfg[dataset].N_SCALES # Escales to extract
  perspective_path = cfg[dataset].PERSPECTIVE_MAP
  is_colored = cfg[dataset].COLOR
      
  return (dataset, use_mask, mask_file, test_names_file, im_folder, 
          dot_ending, pw, pw_norm, sigmadots, n_scales, perspective_path, 
          use_perspective, is_colored, results_file, resize_im)


def dispHelp(arg0):
  print("======================================================")
  print("                       Usage")
  print("======================================================")
  print("\t-h display this message")
  print("\t--cpu_only")
  print("\t--tfclass <Tensorflow class file>")
  print("\t--tfdata <Tensorflow data file>")
  print("\t--tfmodule <Caffe to Tensorflow convertor module")
  print("\t--cfg <config file yaml>")

def main(argv):
    
  # GAME max level
  mx_game = 4 # Max game target

  # Batch size
  b_size = 1

  # CNN vars
  tfdata_path  = 'models/trancos/ccnn/trancos_ccnn.npy'
  tfclass_path = 'models/trancos/ccnn/trancos_ccnn.py'
        
  # Get parameters
  try:
    opts, _ = getopt.getopt(argv, "h:", ["tfclass=", "tfdata=", "tfmodule=",
                                         "cpu_only", "cfg="])
  except getopt.GetoptError as err:
    print("Error while parsing parameters: ", err)
    dispHelp(argv[0])
    return
  
  for opt, arg in opts:
    if opt == '-h':
      dispHelp(argv[0])
      return
    elif opt in ("--tfclass"):
      tfclass_path = arg
    elif opt in ("--tfdata"):
      tfdata_path = arg
    elif opt in ("--tfmodule"):
      tfmodule_path = arg
    elif opt in ("--cpu_only"):
      use_cpu = True
    elif opt in ("--cfg"):
      cfg_file = arg
            
  print("Loading configuration file: ", cfg_file)
  (dataset, use_mask, mask_file, test_names_file, im_folder, 
            dot_ending, pw, pw_norm, sigmadots, n_scales, perspective_path, 
            use_perspective, is_colored, results_file, resize_im) = initTestFromCfg(cfg_file)
            
  print("Choosen parameters:")
  print("-------------------")
  print("Dataset: ", dataset)
  print("Results files: ", results_file)
  print("Test data base location: ", im_folder)
  print("Test inmage names: ", test_names_file)
  print("Dot image ending: ", dot_ending)
  print("Use mask: ", use_mask)
  print("Mask pattern: ", mask_file)
  print("Patch width (pw): ", pw)
  print("Patch width (pw_norm): ", pw_norm)
  print("Sigma for each dot: ", sigmadots)
  print("Number of scales: ", n_scales)
  print("Perspective map: ", perspective_path)
  print("Use perspective:", use_perspective)
  print("TF class path: ", tfclass_path)
  print("TF data path: ", tfdata_path)
  print("TF convertor path: ", tfmodule_path)
  print("Batch size: ", b_size)
  print("Resize images: ", resize_im)
  print("Colored: ", is_colored)
  print("===================")
  
  print("----------------------")
  print("Preparing for Testing")
  print("======================")
    
  print("Reading perspective file")
  if use_perspective:
    pers_file = h5py.File(perspective_path,'r')
    pmap = np.array( pers_file['pmap'] )
    pers_file.close()
      
  mask = None
  if dataset == 'UCSD':
    print("Reading mask")
    if use_mask:
      mask_f = h5py.File(mask_file,'r')
      mask = np.array(mask_f['mask'])
      mask_f.close()
  
  print("Reading image file names:")
  im_names = np.loadtxt(test_names_file, dtype='str')

  # Perform test
  ntrueall=[]
  npredall=[]
  
  # Init GAME
  n_im = len( im_names )
  game_table = np.zeros( (n_im, mx_game) )
  
  # Init CNN
  net, weights = load_model(tfdata_path, tfclass_path, tfmodule_path)
  input_image = net.inputs['data_s0']
  #CNN = TFPredictor(tfdata_path, tfclass_path, tfmodule_path, n_scales)
  
  print 
  print("Start prediction ...")
  count = 0
  gt_vector = np.zeros((len(im_names)))
  pred_vector = np.zeros((len(im_names)))

  with tf.Session() as sess:
    # Load the converted parameters
    sess.run(tf.global_variables_initializer())
    net.load(weights, sess)
  
    for ix, name in enumerate(im_names):
      # Get image paths
      im_path = utl.extendName(name, im_folder)
      dot_im_path = utl.extendName(name, im_folder, use_ending=True, pattern=dot_ending)
      print(name, im_path, dot_im_path)

      # Read image files
      im = loadImage(im_path, color = is_colored)
      dot_im = loadImage(dot_im_path, color = True)
      #print im.shape, dot_im.shape

      # Generate features
      if use_perspective:
        dens_im = genPDensity(dot_im, sigmadots, pmap)
      else:
        dens_im = genDensity(dot_im, sigmadots)
      #print dens_im.shape, dens_im, np.sum(dens_im)
      #imgplot = plt.imshow(dens_im,cmap=cm.jet)
      #plt.colorbar()
      #plt.show()
      
      if resize_im > 0:
        # Resize image
        im = utl.resizeMaxSize(im, resize_im)
        gt_sum = dens_im.sum()
        dens_im = utl.resizeMaxSize(dens_im, resize_im)
        dens_im = dens_im * gt_sum / dens_im.sum()
        
      # Get mask if needed
      if dataset != 'UCSD':
        if use_mask:
          mask_im_path = utl.extendName(name, im_folder, use_ending=True, pattern=mask_file)
          mask = sio.loadmat(mask_im_path, chars_as_strings=1, matlab_compatible=1)
          mask = mask.get('BW')

      start_time=time.time()

      [heith, width] = im.shape[0:2]
      pos = utl.get_dense_pos(heith, width, pw, stride=10)
      #print pos

      # Initialize density matrix and vouting count
      dens_map = np.zeros( (heith, width), dtype = np.float32 )   # Init density to 0
      count_map = np.zeros( (heith, width), dtype = np.int32 )     # Number of votes to divide

      # Iterate for all patches
      for p in pos:
        # Compute displacement from centers
        dx=dy=int(pw/2)

        # Get roi
        x,y=p
        sx=slice(x-dx,x+dx+1,None)
        sy=slice(y-dy,y+dy+1,None)
        crop_im=im[sx,sy,...]
        h, w = crop_im.shape[0:2]
        if h!=w or (h<=0):
            continue
        
        crop_im = utl.resizePatches([crop_im], (pw_norm,pw_norm))
        #print crop_im[0].shape
        
        # Get all the scaled images
        im_scales = extractEscales(crop_im, n_scales)
        #print len(im_scales[0]), im_scales[0][0].shape
        
        # Load and forward CNN
        for s in range(n_scales):          
          pred = sess.run(net.get_output(), feed_dict={input_image: np.expand_dims(im_scales[0][s],0)})

        # Make it squared
        p_side = int(np.sqrt( len( pred.flatten() ) )) 
        pred = pred.reshape(  (p_side, p_side) )
        #print("shape of pred:", pred.shape
        
        # Resize it back to the original size
        pred = utl.resizeDensityPatch(pred, (pw,pw))
        pred[pred<0] = 0
        #print("shape of pred:", pred.shape
        
        # Sumup density map into density map and increase count of votes
        dens_map[sx,sy] += pred
        count_map[sx,sy] += 1

      # Remove Zeros
      count_map[ count_map == 0 ] = 1

      # Average density map
      resImg = dens_map / count_map

      # Mask image if provided
      if mask is not None:
        resImg = resImg  * mask
        gtdots = dens_im * mask

      npred=resImg.sum()
      ntrue=gtdots.sum()

      #ntrue,npred,resImg,gtdots=testOnImg(CNN, im, dens_im, pw, pw_norm, mask)      
      print("image: %d , ntrue = %.2f ,npred = %.2f , time =%.2f sec"%(count,ntrue,npred,time.time()-start_time))
      
      # Keep individual predictions
      gt_vector[ix] = ntrue
      pred_vector[ix] = npred    

      # Hold predictions and originasl
      ntrueall.append(ntrue)
      npredall.append(npred)
        
      # Compute game metric
      for l in range(mx_game):
        game_table[count, l] = gameMetric(resImg, gtdots, l)
    
      count = count +1
          
  ntrueall=np.asarray(ntrueall)
  npredall=np.asarray(npredall)
  print("done ! mean absolute error %.2f" % np.mean(np.abs(ntrueall-npredall)))

  # Print Game results
  results = np.zeros(mx_game)
  for l in range(mx_game):
    results[l] = np.mean( game_table[:,l] )
    print("GAME for level %d: %.2f " % (l, np.mean( game_table[:,l] )))
  
  # Dump results into a txt file
  np.savetxt(results_file + '_pred.txt', npredall)
  np.savetxt(results_file + '_gt.txt', ntrueall)
  
  return 0

if __name__=="__main__":
  main(sys.argv[1:])