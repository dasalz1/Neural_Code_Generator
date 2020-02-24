def check_threads(threads):
	spawn_new = False
	for idx in reversed(range(len(threads))):
		if not threads[idx].is_alive():
			del threads[idx]
			spawn_new = True

	return spawn_new


def threaded_tokenizer(lines, lock, tokens):
	for line in lines:
		line_tokens = tokenize_fine_grained(line)
		for token in line_tokens:
			if token in tokens: continue
			lock.acquire()
			tokens[token] = tokens.get(token, len(tokens))
			lock.release()