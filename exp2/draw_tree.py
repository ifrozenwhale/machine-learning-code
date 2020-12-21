
root = "0"
fontsize = "12"
fontname = "SimSun"


def _sub_plot(g, tree, inc):
    global root

    first_label = list(tree.keys())[0]
    ts = tree[first_label]
    for i in ts.keys():
        if isinstance(tree[first_label][i], dict):
            root = str(int(root) + 1)
            g.node(root, list(tree[first_label][i].keys())[
                   0], fontname=fontname, fontsize=fontsize)
            g.edge(inc, root, str(i), fontname=fontname, fontsize=fontsize)
            _sub_plot(g, tree[first_label][i], root)
        else:
            root = str(int(root) + 1)
            # print(type(tree[first_label][i]))
            g.node(root, tree[first_label][i],
                   fontname=fontname, fontsize=fontsize, style="filled")
            g.edge(inc, root, str(i), fontname=fontname, fontsize=fontsize)


def draw(tree, name, filetype):
    from graphviz import Digraph
    g = Digraph("G", filename=name, format='png', strict=False)
    first_label = list(tree.keys())[0]

    g.node("0", first_label, fontname=fontname, fontsize=fontsize)
    _sub_plot(g, tree, "0")
    g.save()
    import os
    os.system(
        "dot  -T{} {} -Gdpi=600 -o {}.{}".format(filetype, name, name, filetype))
    # g.render('decision.gv', view=False)
