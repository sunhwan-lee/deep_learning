from __future__ import print_function

import json
import argparse
from watson_developer_cloud import ToneAnalyzerV3

import numpy as np
import matplotlib.pyplot as plt

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
  nostalgic_count = 0
  ordinary_emotion = np.array([0.0]*len(emotions))
  ordinary_count = 0

  for idx, t in enumerate(outputs):
    score = [s['score'] for s in t['tone']['tone_categories'][0]['tones']]
    if t['tag'] == 'nostalgic':
      nostalgic_emotion += np.array(score)
      nostalgic_count += 1
    elif t['tag'] == 'ordinary':
      ordinary_emotion += np.array(score)
      ordinary_count += 1
  
  print(emotions)
  print(nostalgic_emotion, nostalgic_count)
  print(ordinary_emotion, ordinary_count)
  
  ind = np.arange(len(emotions))  # the x locations for the groups
  width = 0.35       # the width of the bars
  fig, ax = plt.subplots()
  rects1 = ax.bar(ind, nostalgic_emotion/nostalgic_count, width, color='r')
  rects2 = ax.bar(ind + width, ordinary_emotion/ordinary_count, width, color='y')

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

if __name__ == "__main__":
  args = parse_args()
  #main(args)
  visualize_tone(args)