def to_single_token_stop_ids(tokenizer, stop_token_list):
    if not stop_token_list:
        return []
    ids, seen = [], set()
    for s in stop_token_list:
        if isinstance(s, int):
            tid = [s]
        elif isinstance(s, str):
            tid = tokenizer.encode(s, add_special_tokens=False)
        elif isinstance(s, (list, tuple)) and all(isinstance(x, int) for x in s):
            tid = list(s)
        else:
            continue  
        if len(tid) == 1:
            t = tid[0]
            if t not in seen:
                seen.add(t)
                ids.append(t)
    return ids