require 'numo/narray'

require './embedding'
require './pick_out_words'

N = Numo::DFloat

vocab_file ='./data/glove.6B.50d.txt'
vectors_file ='./data/glove.6B.50d.txt'

# vocab_file ='./data/test_sample.txt'
# vectors_file ='./data/test_sample.txt'

embed = Embedding.new(vocab_file,vectors_file)

cuisine_refs = ['mexican','chinese','french',
                'british','american']
threshold = 0.2

text = 'I want to find an indian restaurant'

cuisines = find_similar_words(embed,text,cuisine_refs,threshold)
puts cuisines
# >>> ['indian']
