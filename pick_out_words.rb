def find_similar_words(embed,text,refs,thresh)
  c = N.zeros(refs.size, embed.w.shape[1])

  refs.map do |idx, term|
    next unless embed.vocab.key?(term)

    c[idx, 0..-1] = embed.w[embed.vocab[term], 0..-1]
  end

  tokens = text.split(' ')

  scores = [0.0] * tokens.size
  found = []
  tokens.each_with_index.map do |term, idx|
    next unless embed.vocab.key?(term)
    vec = embed.w[embed.vocab[term], 0..-1]

    cosines = c.dot(vec.transpose)

    score = cosines.mean
    scores[idx] = score

    found << term if score > thresh
  end

  puts scores

  found
end
