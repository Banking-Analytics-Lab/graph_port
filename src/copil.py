# Ask the user to provide a line of text.
# Scan the text for the following mildly offensive words: \
# arse, bloody, damn, dummy.
# If you find any, then replace its letters with asterisks \
# except for the first letter in each offensive word.
# Print the resulting text.


def main():
    text = input("Enter some text: ")
    text = text.replace("arse", "a**e")
    text = text.replace("bloody", "b****y")
    text = text.replace("damn", "d**n")
    text = text.replace("dummy", "d****y")
    print(text)


main()
