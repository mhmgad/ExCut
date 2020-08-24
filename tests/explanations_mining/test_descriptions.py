from explanations_mining.descriptions_new import Description2, Atom


def test_equal_descriptions():
    d1=Description2(body=[Atom('?x1','created','?x'), Atom('?x','rdf:type','wordnet_book_106410904')], head=Atom('?x','label','book'))
    d2 = Description2(body=[Atom('?x1', 'created', '?x'), Atom('?x', 'rdf:type', 'wordnet_book_106410904')],
                      head=Atom('?x', 'label', 'book'))
    assert(d1==d1)
    assert(d1==d2)
    d3 = Description2(body=[Atom('?x', 'rdf:type', 'wordnet_book_106410904'), Atom('?x1', 'created', '?x')],
                      head=Atom('?x', 'label', 'song'))
    assert(d1==d3)

    d4 = Description2(body=[Atom('?x', 'rdf:type', 'wordnet_book_106410904'), Atom('?y', 'created', '?x')],
                      head=Atom('?x', 'label', 'song'))

    assert (d1 == d4)


def test_not_equal_descriptions():
    d1=Description2(body=[Atom('?x1','created','?x'), Atom('?x','rdf:type','wordnet_book_106410904')], head=Atom('?x','label','book'))
    d2 = Description2(body=[Atom('?x1', 'created', '?x')],
                      head=Atom('?x', 'label', 'book'))
    assert(d1==d1)
    assert(d1!=d2)
    d3 = Description2(body=[Atom('?x', 'rdf:type', '?x4'), Atom('?x1', 'created', '?x')],
                      head=Atom('?x', 'label', 'song'))
    assert(d1!=d3)

    d4 = Description2(body=[Atom('?x', 'rdf:type', '?z'), Atom('?x7', 'created', '?x')],
                      head=Atom('?x', 'label', 'song'))

    assert (d3 == d4)

    d5 = Description2(body=[Atom('?x', 'rdf:type', '?z'), Atom('ahmed', 'created', '?x'),  Atom('mahmoud', 'created', '?x')],
                      head=Atom('?x', 'label', 'song'))
    d6 = Description2(
        body=[ Atom('tuka', 'created', '?x'), Atom('?x', 'rdf:type', '?z'), Atom('mahmoud', 'created', '?x')],
        head=Atom('?x', 'label', 'song'))

    assert(d5!=d6)

def test_descriptions_set():
    ds=set()
    d1 = Description2(body=[Atom('?x1', 'created', '?x'), Atom('?x', 'rdf:type', 'wordnet_book_106410904')],
                      head=Atom('?x', 'label', 'book'))
    ds.add(d1)
    d2 = Description2(body=[Atom('?x1', 'created', '?x')],
                      head=Atom('?x', 'label', 'book'))
    ds.add(d2)
    assert(len(ds)==2)
    d3 = Description2(body=[Atom('?x', 'rdf:type', 'wordnet_book_106410904'), Atom('?x1', 'created', '?x')],
                      head=Atom('?x', 'label', 'song'))

    ds.add(d3)
    assert(len(ds)==2)
    d5 = Description2(
        body=[Atom('?x', 'rdf:type', '?z'), Atom('ahmed', 'created', '?x'), Atom('mahmoud', 'created', '?x')],
        head=Atom('?x', 'label', 'song'))
    ds.add(d5)
    assert(len(ds)==3)

    d6 = Description2(
        body=[Atom('tuka', 'created', '?x'), Atom('?x', 'rdf:type', '?z'), Atom('mahmoud', 'created', '?x')],
        head=Atom('?x', 'label', 'song'))
    ds.add(d6)
    assert(len(ds)==4)

    d4 = Description2(body=[Atom('?x', 'rdf:type', '?z'), Atom('?x7', 'created', '?x')],
                      head=Atom('?x', 'label', 'song'))
    ds.add(d4)
    assert (len(ds) == 5)

    d8= Description2(body=[Atom('?x', 'rdf:type', '?x4'), Atom('?x1', 'created', '?x')],
                      head=Atom('?x', 'label', 'song'))
    ds.add(d8)
    assert (len(ds) == 5)



