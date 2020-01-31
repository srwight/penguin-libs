# File created by Stephen Wight

def demotify(indoc:str) -> str:
    """demotify - remove select emoticons from strings, then return the strings

    This function removes any emoticons (the precursor to graphical emoji) from 
    a string of text, replacing them with a word that represents their emotional
    content.

    Args:
        indoc: This is the string that may contain emoticons to be replaced.

    Returns:
        str: This function returns the string with emoticons replaced by words


    """

    emoticon_dict = {
        ':)': ' happy ',
        ':-)':' happy ',
        ':(': ' sad ',
        ':-(':' sad ',
        '>:(':' angry ',
        ':D': ' grin ',
        ':-D':' grin ',
        ';)': ' wink ',
        ';-)':' wink '
    }

    for key in emoticon_dict:
        indoc = indoc.replace(key, emoticon_dict[key])

    return indoc


if __name__ == '__main__':
    print (demotify('Hey there ;) I\'m :)!'))
