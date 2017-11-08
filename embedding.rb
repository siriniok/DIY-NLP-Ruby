class Embedding
  def initialize(vocab_file, vectors_file)
    words = File.open(vocab_file, 'r') do |f|
      f.readlines.map { |l| l.rstrip.split(' ').first }
    end

    vectors = {}
    File.open(vectors_file, 'r') do |f|
      f.readlines.map do |l|
        vals = l.rstrip.split(' ')
        vectors[vals[0]] = vals[1..-1].map(&:to_f)
      end
    end

    vocab = Hash[words.each_with_index.map { |w, idx| [w, idx] }]
    ivocab = Hash[words.each_with_index.map { |w, idx| [idx, w] }]

    vocab_size = words.size
    vector_dim = vectors[ivocab[0]].size

    w = N.zeros(vocab_size, vector_dim)
    vectors.map do |word, v|
      next if word == '<unk>'

      w[vocab[word], true] = v
    end

    # normalize each word vector to unit variance
    w_norm = N.zeros(w.shape)
    d = (w**2).sum(axis: 1)**(0.5)
    w_norm = (w.transpose / d).transpose

    @w = w_norm
    @vocab = vocab
    @ivocab = ivocab
  end

  attr_reader :w, :vocab, :ivocab
end
