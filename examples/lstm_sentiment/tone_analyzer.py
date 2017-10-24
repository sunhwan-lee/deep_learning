from __future__ import print_function

import json
import argparse
from watson_developer_cloud import ToneAnalyzerV3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from utils.data_manager import DataManager

# Credential of tone analyzer
tone_analyzer = ToneAnalyzerV3(
  username="69c8d6a7-0468-4fff-ba8d-78ecf832d5dc",
  password="cKJqIEb8enDG",
  version='2016-05-19')

# parse commandline arguments
def parse_args():
  '''
  Parses arguments for retrieve lat / long of cities in input file
  '''
  parser = argparse.ArgumentParser()

  parser.add_argument('--data_dir', nargs='?', default="data",
                      help='folder of processed input text data')
  parser.add_argument('--save_fig', action='store_true',
                      help='Save figure')

  return parser.parse_args()

def main(args):

  # Load data
  dm = DataManager(data_dir=args.data_dir)
  
  outputs = []
  #iterate text and run tone analyzer
  for idx,s in enumerate(dm._raw_samples):
    tag = 'ordinary' if dm._sentiments[idx,0] else 'nostalgic'
    print('tag:{0}'.format(tag))
    print('{0} text:{1}\n'.format(idx, s))
    tone_dict = {'text':s,
                 'tag':tag,
                 'tone':tone_analyzer.tone(text=s, tones='emotion', sentences=False)['document_tone']}
    outputs.append(tone_dict)
    
  with open('output/tone_analyzer.json', 'w') as fout:
    json.dump(outputs, fout)

def visualize_tone(args):

  with open('output/tone_analyzer.json') as data_file:
    outputs = json.load(data_file)

  emotions = [e['tone_name'] for e in outputs[0]['tone']['tone_categories'][0]['tones']]
  nostalgic_emotion = np.array([0.0]*len(emotions))
  nostalgic_emotion_lst = []
  nostalgic_count = 0
  ordinary_emotion = np.array([0.0]*len(emotions))
  ordinary_emotion_lst = []
  ordinary_count = 0

  for idx, t in enumerate(outputs):
    score = [s['score'] for s in t['tone']['tone_categories'][0]['tones']]
    if t['tag'] == 'nostalgic':
      nostalgic_emotion += np.array(score)
      nostalgic_emotion_lst.append(score)
      nostalgic_count += 1
    elif t['tag'] == 'ordinary':
      ordinary_emotion += np.array(score)
      ordinary_emotion_lst.append(score)
      ordinary_count += 1
  
  print(emotions)
  print(nostalgic_emotion, nostalgic_count)
  print(ordinary_emotion, ordinary_count)
  
  ind = np.arange(len(emotions))  # the x locations for the groups
  width = 0.35       # the width of the bars
  fig, ax = plt.subplots()
  rects1 = ax.bar(ind, nostalgic_emotion/nostalgic_count, width, color='darkkhaki')
  rects2 = ax.bar(ind + width, ordinary_emotion/ordinary_count, width, color='royalblue')

  # add some text for labels, title and axes ticks
  ax.set_ylabel('Average Tone Score')
  ax.set_title('Emotion Score from Tone Analyzer')
  ax.set_xticks(ind + width / 2)
  ax.set_xticklabels(emotions)

  ax.legend((rects1[0], rects2[0]), ('Nostalgic', 'Ordinary'))
  if args.save_fig:
    plt.savefig('output/tone_analyzer_emotion_comparison.pdf', bbox_inches='tight')
  else:
    plt.show()

  plt.close()

  # create box plot
  nostalgic_emotion_lst = np.array(nostalgic_emotion_lst)
  ordinary_emotion_lst = np.array(ordinary_emotion_lst)
  data_for_box_plot = []
  for i in range(len(emotions)):
    data_for_box_plot.append(nostalgic_emotion_lst[:,i])
    data_for_box_plot.append(ordinary_emotion_lst[:,i])
  
  fig, ax1 = plt.subplots(figsize=(10, 6))
  plt.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)

  bp = plt.boxplot(data_for_box_plot, notch=0, sym='+', vert=1, whis=1.5, showmeans=True)
  plt.setp(bp['boxes'], color='black')
  plt.setp(bp['whiskers'], color='black')
  plt.setp(bp['fliers'], color='red', marker='+')

  # Add a horizontal grid to the plot, but make it very light in color
  # so we can use it for reading data values but not be distracting
  ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                 alpha=0.5)

  # Hide these grid behind plot objects
  ax1.set_axisbelow(True)
  ax1.set_title('Comparison of Emotion Tones')
  ax1.set_xlabel('Emotions')
  ax1.set_ylabel('Scores')

  # Now fill the boxes with desired colors
  boxColors = ['darkkhaki', 'royalblue']
  numBoxes = len(emotions)*2
  medians = list(range(numBoxes))
  for i in range(numBoxes):
    box = bp['boxes'][i]
    boxX = []
    boxY = []
    for j in range(5):
      boxX.append(box.get_xdata()[j])
      boxY.append(box.get_ydata()[j])
    boxCoords = list(zip(boxX, boxY))
    # Alternate between Dark Khaki and Royal Blue
    k = i % 2
    boxPolygon = Polygon(boxCoords, facecolor=boxColors[k])
    ax1.add_patch(boxPolygon)

  # Set the axes ranges and axes labels
  ax1.set_xlim(0.5, numBoxes + 0.5)
  top = 1.2
  bottom = -0.2
  ax1.set_ylim(bottom, top)
  xtickNames = plt.setp(ax1, xticklabels=np.repeat(emotions,2))
  plt.setp(xtickNames, rotation=45, fontsize=10)

  # Finally, add a basic legend
  plt.figtext(0.15, 0.8, 'Nostalgic',
              backgroundcolor=boxColors[0], color='black', weight='roman',
              size='small')
  plt.figtext(0.15, 0.75, 'Ordinary',
              backgroundcolor=boxColors[1],
              color='white', weight='roman', size='small')

  if args.save_fig:
    plt.savefig('output/tone_analyzer_emotion_boxplot.pdf', bbox_inches='tight')
  else:
    plt.show()

  plt.close()
  plt.show()

if __name__ == "__main__":
  args = parse_args()
  #main(args)
  visualize_tone(args)