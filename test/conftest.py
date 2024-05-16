import pytest


@pytest.fixture
def test_data():
    return """3 ||| It 's a lovely film with lovely performances by Buy and Accorsi .
2 ||| No one goes unindicted here , which is probably for the best .
3 ||| And if you 're not nearly moved to tears by a couple of scenes , you 've got ice water in your veins .
4 ||| A warm , funny , engaging film .
4 ||| Uses sharp humor and insight into human nature to examine class conflict , adolescent yearning , the roots of friendship and sexual identity .
2 ||| Half Submarine flick , Half Ghost Story , All in one criminally neglected film
3 ||| Entertains by providing good , lively company .
4 ||| Dazzles with its fully-written characters , its determined stylishness -LRB- which always relates to characters and story -RRB- and Johnny Dankworth 's best soundtrack in years .
4 ||| Visually imaginative , thematically instructive and thoroughly delightful , it takes us on a roller-coaster ride from innocence to experience without even a hint of that typical kiddie-flick sentimentality .
3 ||| Nothing 's at stake , just a twisty double-cross you can smell a mile away -- still , the derivative Nine Queens is lots of fun ."""


@pytest.fixture
def parsed_test_data(test_data):
    return [
        ("3", "it 's a lovely film with lovely performances by buy and accorsi ."),
        ("2", "no one goes unindicted here , which is probably for the best ."),
        (
            "3",
            "and if you 're not nearly moved to tears by a couple of scenes , you 've got ice water in your veins .",
        ),
        ("4", "a warm , funny , engaging film ."),
        (
            "4",
            "uses sharp humor and insight into human nature to examine class conflict , adolescent yearning , the roots of friendship and sexual identity .",
        ),
        (
            "2",
            "half submarine flick , half ghost story , all in one criminally neglected film",
        ),
        ("3", "entertains by providing good , lively company ."),
        (
            "4",
            "dazzles with its fully-written characters , its determined stylishness -lrb- which always relates to characters and story -rrb- and johnny dankworth 's best soundtrack in years .",
        ),
        (
            "4",
            "visually imaginative , thematically instructive and thoroughly delightful , it takes us on a roller-coaster ride from innocence to experience without even a hint of that typical kiddie-flick sentimentality .",
        ),
        (
            "3",
            "nothing 's at stake , just a twisty double-cross you can smell a mile away -- still , the derivative nine queens is lots of fun .",
        ),
    ]
