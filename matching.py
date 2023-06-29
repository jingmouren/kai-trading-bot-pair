from utils import *

__author__ = 'kq'


class Signal:

    def __init__(self):
        self.max_cardinality = 1
        self.min_p = 0.05


def update_partnership(partnership: pd.DataFrame) -> pd.DataFrame:
    """

    :param partnership:
    :return:
    """

    # Count matches per asset
    feasible_count = pd.concat([partnership.ASSET_1.value_counts(), partnership.ASSET_2.value_counts()], axis=0)
    good_partners = feasible_count.groupby(feasible_count.index).sum().sort_values()
    partnership['ASSET_1_PARTNERS'] = partnership.ASSET_1.map(good_partners.to_dict())
    partnership['ASSET_2_PARTNERS'] = partnership.ASSET_2.map(good_partners.to_dict())

    # Find least/most matched assets
    partnership['MINIMUM_MATCHES'] = partnership[['ASSET_1_PARTNERS', 'ASSET_2_PARTNERS']].min(axis=1)
    partnership = partnership.sort_values('MINIMUM_MATCHES').reset_index(drop=True)
    partnership['MINIMUM_INDEX'] = pd.Series(np.where(partnership.ASSET_1_PARTNERS == partnership.MINIMUM_MATCHES,
                                                      partnership.ASSET_1, partnership.ASSET_2),
                                             index=partnership.index)
    partnership['MAXIMUM_INDEX'] = pd.Series(np.where(partnership.ASSET_1_PARTNERS == partnership.MINIMUM_MATCHES,
                                                      partnership.ASSET_2, partnership.ASSET_1),
                                             index=partnership.index)
    return partnership.sort_values('MINIMUM_MATCHES').reset_index(drop=True)


def approx_maximal_matching(partnership: pd.DataFrame) -> List[str]:
    """

    :param partnership:
    :return:
    """
    feasible_set, tracker = list(set(partnership.MINIMUM_INDEX)), partnership.copy()
    optimal_pairs = []
    while len(feasible_set) > 0:

        partnership = update_partnership(partnership=partnership)

        # Get next ticker
        ticker = feasible_set[0]
        try:

            # Pairing not possible
            if ticker not in list(partnership.ASSET_1) + list(partnership.ASSET_2):
                feasible_set.remove(ticker)
                continue

            # Get current matching
            partnership = update_partnership(partnership=partnership)

            # Check pair availability
            pair = partnership[partnership.MINIMUM_INDEX == ticker]
            if len(pair) < 1:
                feasible_set.remove(ticker)
                continue

            else:

                # Assign
                pair = pair.iloc[0].NAME
                partner = partnership[partnership.MINIMUM_INDEX == ticker].iloc[0].MAXIMUM_INDEX
                optimal_pairs.append(pair)

                # Update feasible set
                if ticker in list(feasible_set):
                    feasible_set.remove(ticker)
                if partner in list(feasible_set):
                    feasible_set.remove(partner)

                    # Prevent repeats
                partnership = partnership[(partnership.ASSET_1 != ticker) & (partnership.ASSET_2 != ticker)]
                partnership = partnership[(partnership.ASSET_1 != partner) & (partnership.ASSET_2 != partner)]

        except IndexError:
            pass
        print('Matched {} and {}. Current length of feasible set: {}'.format(ticker, partner, len(feasible_set)))
    return optimal_pairs


def draw_graph(signal_data: pd.DataFrame,
               method: str,
               date: str) -> None:
    assert (method in ['max_weight', 'maximal', 'min_max', 'baseline', 'approx_maximal'])
    edges = [tuple(x) for x in signal_data[['ASSET_1', 'ASSET_2', 'T_STAT']].values.tolist()]
    G = nx.Graph()
    G.add_nodes_from(signal_data['ASSET_1'].unique(), bipartite=0, label='ASSET_1')
    G.add_nodes_from(signal_data['ASSET_2'].unique(), bipartite=1, label='ASSET_2')

    # Now assign the ratings correctly to edges
    for row in edges:
        G.add_edge(row[0], row[1], rating=row[2])
    left_or_top = signal_data['ASSET_1'].unique()

    # Draw the graph
    pos = nx.bipartite_layout(G, left_or_top)
    plt.figure(3, figsize=(50, 50))
    nx.draw(G,
            pos,
            node_color='#A0CBE2',
            edge_color='#00bb5e',
            width=1,
            edge_cmap=plt.cm.Blues, with_labels=True)

    # Get the edge labels for ratings
    edge_labels = nx.get_edge_attributes(G, 'rating')

    # Draw the edge labels
    graph = nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.savefig(HOME + '/{method}/figures/{date}.png'.format(method=method, date=date.replace('-', '')))
    plt.clf()


def store_matchings() -> None:
    """

    :return:
    """
    # Get ADF test history
    adf_files = sorted(glob.glob(HOME + '/adf/*.csv'))
    for file in adf_files:
        date = file.split('/')[-1].split('.')[0]
        data = pd.read_csv(file).sort_values('T_STAT')
        data.T_STAT = data.T_STAT.round(2)
        data['TUPLE'] = list(zip(data.ASSET_1, data.ASSET_2))
        print('Found {} possible pairs for {}'.format(len(data), date))

        # Create bipartite graph
        G = nx.Graph()
        G.add_nodes_from(data['ASSET_1'].unique(), bipartite=0, label='ASSET_1')
        G.add_nodes_from(data['ASSET_2'].unique(), bipartite=1, label='ASSET_2')

        # Now assign the ratings correctly to edges
        edges = [tuple(x) for x in data[['ASSET_1', 'ASSET_2', 'T_STAT']].values.tolist()]
        for row in edges:
            G.add_edge(row[0], row[1], rating=row[2])

        # Baseline
        baseline = set(data[data.P_VALUE <= Signal().min_p].TUPLE)
        graph = data[data.TUPLE.isin(max_weight)]
        draw_graph(signal_data=graph, method='min_max', date=date)

        # Max weight matching
        max_weight = max_weight_matching(G, maxcardinality=Signal().max_cardinality)
        graph = data[data.TUPLE.isin(max_weight)]
        draw_graph(signal_data=graph, method='max_weight', date=date)

        # Maximal matching
        maximal = maximal_matching(G)
        graph = data[data.TUPLE.isin(max_weight)]
        draw_graph(signal_data=graph, method='maximal', date=date)

        # Approx maximal
        approx_maximal = approx_maximal_matching(adf=data)
        graph = data[data.TUPLE.isin(approx_maximal)]
        draw_graph(signal_data=graph, method='approx_maximal', date=date)

        # Store optimal pairs
        optimal_pairs = pd.Series({'BASELINE': list(baseline),
                                   'MAX_WEIGHT': list(max_weight),
                                   'MAXIMAL': list(maximal),
                                   'APPROX_MAXIMAL': list(optimal_pairs)}
                                  ).to_frame().reset_index().rename(columns={0: 'PAIRS', 'index': 'METHOD'})
        optimal_pairs['NUM_PAIRS'] = optimal_pairs.PAIRS.str.len()
        optimal_pairs.to_csv(HOME + 'optimal_pairs/{}.csv'.format(date.replace('-', '')), index=False)