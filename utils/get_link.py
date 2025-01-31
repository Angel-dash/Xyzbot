def get_source_link(metadatas):
    link = 'https://www.youtube.com/watch?v='
    yt_link = []
    for metadata in metadatas:
        source = metadata['source']
        values = source.split('.txt')

        link = link + values[0]
        yt_link.append(link)
        # print(yt_link)
    return yt_link