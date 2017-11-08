def find_similar_words(embed, text, refs, thresh)
  c = N.zeros(refs.size, embed.w.shape[1])

  refs.each_with_index.map do |term, idx|
    next unless embed.vocab.key?(term)

    c[idx, true] = embed.w[embed.vocab[term], true]
  end

  tokens = text.split(' ')

  scores = [0.0] * tokens.size
  found = []
  tokens.each_with_index.map do |term, idx|
    next unless embed.vocab.key?(term)

    vec = embed.w[embed.vocab[term], true]
    cosines = c.dot(vec.transpose)
    score = cosines.mean
    scores[idx] = score

    found << term if score > thresh
  end

  puts scores

  found
end
