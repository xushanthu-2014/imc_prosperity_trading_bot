import json
from datamodel import Order, ProsperityEncoder, Symbol, TradingState, OrderDepth
from typing import Any, Dict, List

# pair trading params
# pair 1 
# fair value = 1000
# edge = +11 -10

# pair 2
# fair value = 400
# edge = +40 -30
import json
from datamodel import Order, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any

class Logger:
    def __init__(self) -> None:
        self.logs = ""

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]]) -> None:
        print(json.dumps({
            "state": self.compress_state(state),
            "orders": self.compress_orders(orders),
            "logs": self.logs,
        }, cls=ProsperityEncoder, separators=(",", ":"), sort_keys=True))

        self.logs = ""

    def compress_state(self, state: TradingState) -> dict[str, Any]:
        listings = []
        for listing in state.listings.values():
            listings.append([listing["symbol"], listing["product"], listing["denomination"]])

        order_depths = {}
        for symbol, order_depth in state.order_depths.items():
            order_depths[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return {
            "t": state.timestamp,
            "l": listings,
            "od": order_depths,
            "ot": self.compress_trades(state.own_trades),
            "mt": self.compress_trades(state.market_trades),
            "p": state.position,
            "o": state.observations,
        }

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append([
                    trade.symbol,
                    trade.buyer,
                    trade.seller,
                    trade.price,
                    trade.quantity,
                    trade.timestamp,
                ])

        return compressed

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

logger = Logger()


class Trader:
    # define data members
    def __init__(self):
        self.position_limit = {"PEARLS": 20, "BANANAS": 20, 'COCONUTS': 600, 'PINA_COLADAS': 300, 'BERRIES': 250,
                               'DIVING_GEAR': 50, 'DOLPHIN_SIGHTINGS': 0, 'PICNIC_BASKET': 70, 'UKULELE': 70, 'DIP': 300, 'BAGUETTE': 150}
        self.last_mid_price = {'PEARLS': 10000, 'BANANAS': 4900, 'COCONUTS': 8000, 'PINA_COLADAS': 15000, 'BERRIES': 4000,
                               'DIVING_GEAR': 10000, 'DOLPHIN_SIGHTINGS': 0, 'PICNIC_BASKET': 73000, 'UKULELE': 21000, 'DIP': 7000, 'BAGUETTE': 12000}
        self.acceptable_price = {'PEARLS': 10000, 'BANANAS': 5000, 'COCONUTS': 8000, 'PINA_COLADAS': 15000, 'BERRIES': 4000,
                                 'DIVING_GEAR': 10000, 'DOLPHIN_SIGHTINGS': 0, 'PICNIC_BASKET': 73000, 'UKULELE': 21000, 'DIP': 7000, 'BAGUETTE': 12000}
        self.legal_buy_vol = {'PEARLS': 20, 'BANANAS': 20, 'COCONUTS': 600, 'PINA_COLADAS': 300, 'BERRIES': 250,
                              'DIVING_GEAR': 50, 'DOLPHIN_SIGHTINGS': 0, 'PICNIC_BASKET': 70, 'UKULELE': 70, 'DIP': 300, 'BAGUETTE': 150}
        self.legal_sell_vol = {'PEARLS': -20, 'BANANAS': -20, 'COCONUTS': -600, 'PINA_COLADAS': -300, 'BERRIES': -250,
                               'DIVING_GEAR': -50, 'DOLPHIN_SIGHTINGS': 0, 'PICNIC_BASKET': -70, 'UKULELE': -70, 'DIP': -300, 'BAGUETTE': -150}
        self.current_position = {'PEARLS': 0, 'BANANAS': 0, 'COCONUTS': 0, 'PINA_COLADAS': 0, 'BERRIES': 0,
                                 'DIVING_GEAR': 0, 'DOLPHIN_SIGHTINGS': 0, 'PICNIC_BASKET': 0, 'UKULELE': 0, 'DIP': 0, 'BAGUETTE': 0}
        self.last_sign = 0
        self.signal_expire =0

        # store the orderbook depth 1 for each product
        self.best_bid = {'PEARLS': 0, 'BANANAS': 0, 'COCONUTS': 0, 'PINA_COLADAS': 0, 'BERRIES': 0,
                         'DIVING_GEAR': 0, 'DOLPHIN_SIGHTINGS': 0, 'PICNIC_BASKET': 0, 'UKULELE': 0, 'DIP': 0, 'BAGUETTE': 0}
        self.best_ask = {'PEARLS': 0, 'BANANAS': 0, 'COCONUTS': 0, 'PINA_COLADAS': 0, 'BERRIES': 0,
                         'DIVING_GEAR': 0, 'DOLPHIN_SIGHTINGS': 0, 'PICNIC_BASKET': 0, 'UKULELE': 0, 'DIP': 0, 'BAGUETTE': 0}
        self.best_bid_volume = {'PEARLS': 0, 'BANANAS': 0, 'COCONUTS': 0, 'PINA_COLADAS': 0, 'BERRIES': 0,
                                'DIVING_GEAR': 0, 'DOLPHIN_SIGHTINGS': 0, 'PICNIC_BASKET': 0, 'UKULELE': 0, 'DIP': 0, 'BAGUETTE': 0}
        self.best_ask_volume = {'PEARLS': 0, 'BANANAS': 0, 'COCONUTS': 0, 'PINA_COLADAS': 0, 'BERRIES': 0,
                                'DIVING_GEAR': 0, 'DOLPHIN_SIGHTINGS': 0, 'PICNIC_BASKET': 0, 'UKULELE': 0, 'DIP': 0, 'BAGUETTE': 0}
        self.imbalance_threshold = {'PEARLS': 15, 'BANANAS': 15,
                          'BERRIES': 30, 'PICNIC_BASKET': 9}
        self.ukulele_valid_pos = 0
        self.olivia_signal_buy = {'PEARLS': 0, 'BANANAS': 0, 'COCONUTS': 0, 'PINA_COLADAS': 0, 'BERRIES': 0,
                         'DIVING_GEAR': 0, 'DOLPHIN_SIGHTINGS': 0, 'PICNIC_BASKET': 0, 'UKULELE': 0, 'DIP': 0, 'BAGUETTE': 0}
        self.olivia_signal_sell = {'PEARLS': 0, 'BANANAS': 0, 'COCONUTS': 0, 'PINA_COLADAS': 0, 'BERRIES': 0,
                         'DIVING_GEAR': 0, 'DOLPHIN_SIGHTINGS': 0, 'PICNIC_BASKET': 0, 'UKULELE': 0, 'DIP': 0, 'BAGUETTE': 0}

        self.oliva_berry_buy_time = -99

    def market_status(self,product):
        if product not in self.imbalance_threshold.keys():
            return 0
        else:
            N = self.imbalance_threshold[product]
            if self.best_bid_volume[product]!=0 and self.best_ask_volume[product]!=0:
                best_ask_volume = self.best_ask_volume[product]
                best_bid_volume = self.best_bid_volume[product]
                if -best_ask_volume > N and best_bid_volume <= N:
                    return -1
                elif -best_ask_volume <= N and best_bid_volume > N:
                    return 1
                else:
                    return 0
                
    def gen_diving_sig(self,state: TradingState,N = 900):
        dol_sights = state.observations['DOLPHIN_SIGHTINGS']

        dol_diff = dol_sights - self.last_mid_price['DOLPHIN_SIGHTINGS'] if self.last_mid_price['DOLPHIN_SIGHTINGS']!=0 else 0

        instant_signal = 1 if dol_diff > 5 else -1 if dol_diff < -5 else 0

        # memory instant signal for 900 steps
        if instant_signal != 0:
            self.last_sign = instant_signal
            self.signal_expire = state.timestamp+N*100
        else:
            if state.timestamp > self.signal_expire:
                self.last_sign = 0

        self.last_mid_price['DOLPHIN_SIGHTINGS'] = dol_sights

        return self.last_sign
    
    def pair_trading_1(self,state: TradingState) -> tuple[List[Order], List[Order]]:
        """
        Pair trading strategy for COCONUTS and PINA_COLADAS
        :param state: TradingState
        :return (List[Order], List[Order]): C_orders, PC_orders
        """
        C_pos = self.current_position['COCONUTS']
        PC_pos = self.current_position['PINA_COLADAS']
        C_mid, C_bid_1, C_ask_1, C_bidVol_1, C_askVol_1 = self.last_mid_price['COCONUTS'], self.best_bid['COCONUTS'], self.best_ask['COCONUTS'], self.best_bid_volume['COCONUTS'], self.best_ask_volume['COCONUTS']
        PC_mid, PC_bid_1, PC_ask_1, PC_bidVol_1, PC_askVol_1 = self.last_mid_price['PINA_COLADAS'], self.best_bid['PINA_COLADAS'], self.best_ask['PINA_COLADAS'], self.best_bid_volume['PINA_COLADAS'], self.best_ask_volume['PINA_COLADAS']

        C_orders: list[Order] = []
        PC_orders: list[Order] = []

        ETF_price = 2 * C_mid - PC_mid - 1000

        if ETF_price < - 10:
            C_target_pos = int(min(-ETF_price, 20) / 20 * self.position_limit['COCONUTS'])
            PC_target_pos = int(min(-ETF_price, 20) / 20 * (-self.position_limit['PINA_COLADAS']))

            C_volume = min(max(C_target_pos - C_pos, 0), -C_askVol_1)
            PC_volume = max(min(PC_target_pos - PC_pos, 0), -PC_bidVol_1)
            volume = min(C_volume // 2, - PC_volume)
            C_volume = 2 * volume
            PC_volume = - volume

            # Buy COCONUTS
            if volume != 0:
                C_orders.append(Order('COCONUTS', C_ask_1, C_volume))  # market order

            # Sell PINA_COLADAS
            if volume != 0:
                PC_orders.append(Order('PINA_COLADAS', PC_bid_1, PC_volume))  # market order

        elif ETF_price > 11:
            C_target_pos = int(min(ETF_price, 20) / 20 * (-self.position_limit['COCONUTS']))
            PC_target_pos = int(min(ETF_price, 20) / 20 * self.position_limit['PINA_COLADAS'])

            C_volume = max(min(C_target_pos - C_pos, 0), -C_bidVol_1)
            PC_volume = min(max(PC_target_pos - PC_pos, 0), -PC_askVol_1)
            volume = min((-C_volume) // 2, PC_volume)
            C_volume = - 2 * volume
            PC_volume = volume



            # Sell COCONUTS
            if volume != 0:
                C_orders.append(Order('COCONUTS', C_bid_1, C_volume))  # market order

            # Buy PINA_COLADAS
            if volume != 0:
                PC_orders.append(Order('PINA_COLADAS', PC_ask_1, PC_volume))  # market order
        

        return C_orders, PC_orders
    

    def pair_trading_2(self,state: TradingState) -> tuple[List[Order], List[Order], List[Order], List[Order]]:
        # 'BAGUETTE': 150, 'DIP': 300, 'UKULELE': 70, 'PICNIC_BASKET': 70
        P_pos = self.current_position['PICNIC_BASKET']
        B_pos = self.current_position['BAGUETTE']
        D_pos = self.current_position['DIP']
        U_pos = self.current_position['UKULELE']


        P_mid, P_bid_1, P_ask_1, P_bidVol_1, P_askVol_1 = self.last_mid_price['PICNIC_BASKET'], self.best_bid['PICNIC_BASKET'], self.best_ask['PICNIC_BASKET'], self.best_bid_volume['PICNIC_BASKET'], self.best_ask_volume['PICNIC_BASKET']
        B_mid, B_bid_1, B_ask_1, B_bidVol_1, B_askVol_1 = self.last_mid_price['BAGUETTE'], self.best_bid['BAGUETTE'], self.best_ask['BAGUETTE'], self.best_bid_volume['BAGUETTE'], self.best_ask_volume['BAGUETTE']
        D_mid, D_bid_1, D_ask_1, D_bidVol_1, D_askVol_1 = self.last_mid_price['DIP'], self.best_bid['DIP'], self.best_ask['DIP'], self.best_bid_volume['DIP'], self.best_ask_volume['DIP']
        U_mid, U_bid_1, U_ask_1, U_bidVol_1, U_askVol_1 = self.last_mid_price['UKULELE'], self.best_bid['UKULELE'], self.best_ask['UKULELE'], self.best_bid_volume['UKULELE'], self.best_ask_volume['UKULELE']

        P_orders: list[Order] = []
        B_orders: list[Order] = []
        D_orders: list[Order] = []
        U_orders: list[Order] = []

        picnic_etf = P_mid - 2 * B_mid - 4 * D_mid - U_mid - 400

        if len(state.market_trades['UKULELE']) != 0:
            for trade in state.market_trades['UKULELE']:
                if trade.buyer == 'Olivia':
                    self.olivia_signal_buy['UKULELE'] = 1
                if trade.seller == 'Olivia':
                    self.olivia_signal_sell['UKULELE'] = 1

        if self.olivia_signal_buy['UKULELE'] == 0 and self.olivia_signal_sell['UKULELE'] == 0:
            if picnic_etf < - 30:
                # Buy picnic_etf, buy PICNIC_BASKET, sell BAGUETTE DIP UKULELE
                P_target_pos = int(min(-picnic_etf, 200) / 200 * self.position_limit['PICNIC_BASKET'])
                B_target_pos = int(min(-picnic_etf, 200) / 200 * (-self.position_limit['BAGUETTE']))
                D_target_pos = int(min(-picnic_etf, 200) / 200 * (-self.position_limit['DIP']))
                U_target_pos = int(min(-picnic_etf, 200) / 200 * (-self.position_limit['UKULELE']))

                P_volume = min(max(P_target_pos - P_pos, 0), -P_askVol_1)
                B_volume = max(min(B_target_pos - B_pos, 0), -B_bidVol_1)
                D_volume = max(min(D_target_pos - D_pos, 0), -D_bidVol_1)
                #如果-满仓，只卖别的asset
                if U_target_pos - U_pos >= 0:
                    U_volume = max(min(U_target_pos - self.ukulele_valid_pos, 0), -U_bidVol_1)
                #不满仓，正常配平etf
                else:
                    U_volume = max(min(U_target_pos - U_pos, 0), -U_bidVol_1)


                volume = min(P_volume, -B_volume // 2, -D_volume // 4, -U_volume)
                P_volume = volume
                B_volume = - 2 * volume
                D_volume = - 4 * volume
                U_volume = - volume

                if volume != 0:
                    # Buy picnic_etf, buy PICNIC_BASKET, sell BAGUETTE DIP UKULELE
                    P_orders.append(Order('PICNIC_BASKET', P_ask_1, P_volume))
                    B_orders.append(Order('BAGUETTE', B_bid_1, B_volume))
                    D_orders.append(Order('DIP', D_bid_1, D_volume))
                    U_orders.append(Order('UKULELE', U_bid_1, U_volume))
                    self.ukulele_valid_pos += U_volume
                    return P_orders, B_orders, D_orders, U_orders

            elif picnic_etf > 40:
                # Sell picnic_etf, sell PICNIC_BASKET, buy BAGUETTE DIP UKULELE
                P_target_pos = int(min(picnic_etf, 200) / 200 * (-self.position_limit['PICNIC_BASKET']))
                B_target_pos = int(min(picnic_etf, 200) / 200 * self.position_limit['BAGUETTE'])
                D_target_pos = int(min(picnic_etf, 200) / 200 * self.position_limit['DIP'])
                U_target_pos = int(min(picnic_etf, 200) / 200 * self.position_limit['UKULELE'])

                P_volume = max(min(P_target_pos - P_pos, 0), -P_bidVol_1)
                B_volume = min(max(B_target_pos - B_pos, 0), -B_askVol_1)
                D_volume = min(max(D_target_pos - D_pos, 0), -D_askVol_1)
                #如果+满仓，只调别的asset
                if U_target_pos - U_pos <= 0:
                    U_volume = min(max(U_target_pos - self.ukulele_valid_pos, 0), -U_askVol_1)
                #不满仓，正常配平etf
                else:
                    U_volume = min(max(U_target_pos - U_pos, 0), -U_askVol_1)

                volume = min(-P_volume, B_volume // 2, D_volume // 4, U_volume)
                P_volume = - volume
                B_volume = 2 * volume
                D_volume = 4 * volume
                U_volume = volume

                if volume != 0:
                    # Sell picnic_etf, sell PICNIC_BASKET, buy BAGUETTE DIP UKULELE
                    P_orders.append(Order('PICNIC_BASKET', P_bid_1, P_volume))
                    B_orders.append(Order('BAGUETTE', B_ask_1, B_volume))
                    D_orders.append(Order('DIP', D_ask_1, D_volume))
                    U_orders.append(Order('UKULELE', U_ask_1, U_volume))
                    #更新etf相关仓位
                    self.ukulele_valid_pos += U_volume
                    return P_orders, B_orders, D_orders, U_orders
                    #最高点，ukulele
            
        elif self.olivia_signal_buy['UKULELE'] == 1 and self.olivia_signal_sell['UKULELE'] == 0:            

            #最低点，买ukulele

            volume = min(max(self.position_limit['UKULELE'] - U_pos, 0), -U_askVol_1)
            #只满仓ukulele

            U_orders.append(Order('UKULELE', U_ask_1, volume))   
            return P_orders, B_orders, D_orders, U_orders

        elif self.olivia_signal_buy[product] == 0 and self.olivia_signal_sell[product] == 1:
            volume = max(min(-self.position_limit['UKULELE'] - U_pos, 0), -U_bidVol_1)
            U_orders.append(Order('UKULELE', U_bid_1, volume))
            return P_orders, B_orders, D_orders, U_orders
        else:
            if self.current_position['UKULELE'] < 0:
                best_ask = self.best_ask['UKULELE']
                best_ask_volume = self.best_ask_volume['UKULELE']
                U_orders.append(Order('UKULELE', best_ask, int(min(-best_ask_volume, -state.position['UKULELE']))))
            elif self.current_position['UKULELE'] > 0:
                best_bid = self.best_bid[product]
                best_bid_volume = self.best_bid_volume['UKULELE']
                U_orders.append(Order(product, best_bid, int(max(-best_bid_volume, -state.position['UKULELE']))))
            else:
                self.olivia_signal_buy['UKULELE'] = 0
                self.olivia_signal_sell['UKULELE'] = 0
        return P_orders, B_orders, D_orders, U_orders

        

    def copy_trading(self,product,state: TradingState):
        if len(state.market_trades[product]) != 0:
            orders: list[Order] = []
            for trade in state.market_trades[product]:
                if trade.buyer == self.copy_trader[product]:
                    orders.append(Order(trade.symbol, trade.price, int(min(trade.quantity, self.legal_buy_vol[product]))))
                elif trade.seller == self.copy_trader[product]:
                    orders.append(Order(trade.symbol, trade.price, int(max(-trade.quantity, self.legal_sell_vol[product]))))
            return orders

    def run(self,state:TradingState)->Dict[str, List[Order]]:

        result = {}

        # read orderbook informantion and store them in __init__ function
        for product in state.order_depths.keys():
            current_position = 0 if product not in state.position.keys() else state.position[product]

            legal_buy_vol = self.position_limit[product] - current_position
            legal_sell_vol = -self.position_limit[product] - current_position
        
            # orderbook information and mid price
            order_depth: OrderDepth = state.order_depths[product]


            if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
                best_ask = min(order_depth.sell_orders.keys())
                best_ask_volume = order_depth.sell_orders[best_ask]
                best_bid = max(order_depth.buy_orders.keys())
                best_bid_volume = order_depth.buy_orders[best_bid]
                avg_buy_price = sum([order_depth.buy_orders[i] * i for i in order_depth.buy_orders.keys()])/sum(
                    [order_depth.buy_orders[i] for i in order_depth.buy_orders.keys()])
                avg_sell_price = sum([-order_depth.sell_orders[i] * i for i in order_depth.sell_orders.keys()])/sum(
                    [-order_depth.sell_orders[i] for i in order_depth.sell_orders.keys()])
                # mid_price = (avg_sell_price + avg_buy_price)/2 weighted bid weighted ask/2
                buy_order_volume = sum([order_depth.buy_orders[i]
                                    for i in order_depth.buy_orders.keys()])
                sell_order_volume = sum(
                    [-order_depth.sell_orders[i] for i in order_depth.sell_orders.keys()])
                mid_price = (avg_sell_price * buy_order_volume + avg_buy_price *
                            sell_order_volume)/(buy_order_volume + sell_order_volume)

            elif len(order_depth.sell_orders) == 0 and len(order_depth.buy_orders) != 0:  # 只有买单
                best_ask = 0
                best_ask_volume = 0
                best_bid = max(order_depth.buy_orders.keys())
                best_bid_volume = order_depth.buy_orders[best_bid]
                mid_price = self.last_mid_price[product]
            elif len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) == 0:  # 只有卖单
                best_ask = min(order_depth.sell_orders.keys())
                best_ask_volume = order_depth.sell_orders[best_ask]
                best_bid = 0
                best_bid_volume = 0
                mid_price = self.last_mid_price[product]
            else:
                mid_price = self.last_mid_price[product]

            # save the calculation result
            self.best_bid[product] = best_bid
            self.best_ask[product] = best_ask
            self.best_bid_volume[product] = best_bid_volume
            self.best_ask_volume[product] = best_ask_volume
            self.current_position[product] = current_position
            self.legal_buy_vol[product] = legal_buy_vol
            self.legal_sell_vol[product] = legal_sell_vol
            self.last_mid_price[product] = mid_price


        

        for product in state.order_depths.keys():
        # the relationship between coconut and pina colada linear with intercept, 2*coconut = pina colada+1000

            if product == 'PEARLS':
                market_status = self.market_status(product)
                self.acceptable_price[product] = self.last_mid_price[product] + \
                    2*market_status
            elif product == 'BANANAS':
                market_status = self.market_status(product)
                self.acceptable_price[product] = self.last_mid_price[product] + \
                    1*market_status
            
            acceptable_price = self.acceptable_price[product]
            legal_buy_vol = self.legal_buy_vol[product]
            legal_sell_vol = self.legal_sell_vol[product]

            if product == 'BERRIES':
                orders: list[Order] = []

                berries_trades = state.market_trades['BERRIES']
                for trade in berries_trades:
                    if trade.buyer == 'Olivia':
                        self.olivia_signal_buy['BERRIES'] = 1
                        self.oliva_berry_buy_time = state.timestamp
                    if trade.seller == 'Olivia':
                        self.olivia_signal_sell['BERRIES'] = 1

                if self.olivia_signal_buy['BERRIES'] == 1:
                    if legal_buy_vol > 0 and self.best_ask_volume[product] != 0:
                        if self.oliva_berry_buy_time >= 580000:
                            best_ask = self.best_ask[product]
                            best_ask_volume = self.best_ask_volume[product]
                            orders.append(Order(product, best_ask, int(
                                min(-best_ask_volume, -state.position[product]))))
                            legal_buy_vol = legal_buy_vol - int(
                                min(-best_ask_volume, -state.position[product]))
                            legal_sell_vol = legal_sell_vol - - int(
                                min(-best_ask_volume, -state.position[product]))
                        else:
                            best_ask = self.best_ask[product]
                            best_ask_volume = self.best_ask_volume[product]
                            orders.append(Order(product, best_ask, int(
                                min(-best_ask_volume, legal_buy_vol))))
                            legal_sell_vol = legal_sell_vol - \
                                int(min(-best_ask_volume, legal_buy_vol))
                            legal_buy_vol = legal_buy_vol - \
                                int(min(-best_ask_volume, legal_buy_vol))
                    elif legal_buy_vol == 0:
                        self.olivia_signal_buy['BERRIES'] = 0
                    result[product] = orders
                elif self.olivia_signal_sell['BERRIES'] == 1:
                    if legal_sell_vol < 0 and self.best_bid_volume[product] != 0:
                        best_bid = self.best_bid[product]
                        best_bid_volume = self.best_bid_volume[product]
                        orders.append(Order(product, best_bid, int(
                            max(-best_bid_volume, legal_sell_vol))))
                        legal_buy_vol = legal_buy_vol - \
                            int(max(-best_bid_volume, legal_sell_vol))
                        legal_sell_vol = legal_sell_vol - \
                            int(max(-best_bid_volume, legal_sell_vol))
                    elif legal_sell_vol == 0:
                        self.olivia_signal_sell['BERRIES'] = 0
                    result[product] = orders
                else:

                    if state.timestamp >= 250000 and state.timestamp < 280000:
                        if legal_buy_vol > 0 and self.best_ask_volume[product] != 0:
                            best_ask = self.best_ask[product]
                            best_ask_volume = self.best_ask_volume[product]
                            orders.append(Order(product, best_ask, int(
                                min(-best_ask_volume, legal_buy_vol))))
                            legal_sell_vol = legal_sell_vol - \
                                int(min(-best_ask_volume, legal_buy_vol))
                            legal_buy_vol = legal_buy_vol - \
                                int(min(-best_ask_volume, legal_buy_vol))
                        result[product] = orders
                    elif state.timestamp >= 550000 and state.timestamp < 580000:
                        if legal_sell_vol < 0 and self.best_bid_volume[product] != 0:
                            best_bid = self.best_bid[product]
                            best_bid_volume = self.best_bid_volume[product]
                            orders.append(Order(product, best_bid, int(
                                max(-best_bid_volume, legal_sell_vol))))
                            legal_buy_vol = legal_buy_vol - \
                                int(max(-best_bid_volume, legal_sell_vol))
                            legal_sell_vol = legal_sell_vol - \
                                int(max(-best_bid_volume, legal_sell_vol))
                        result[product] = orders
                    elif state.timestamp >= 750000 and state.timestamp < 780000:
                        if state.position[product] < 0:
                            best_ask = self.best_ask[product]
                            best_ask_volume = self.best_ask_volume[product]
                            orders.append(Order(product, best_ask, int(
                                min(-best_ask_volume, -state.position[product]))))
                            legal_buy_vol = legal_buy_vol - int(
                                min(-best_ask_volume, -state.position[product]))
                            legal_sell_vol = legal_sell_vol - - int(
                                min(-best_ask_volume, -state.position[product]))
                        result[product] = orders
                    

            elif product == 'DIVING_GEAR':
                last_sign = self.gen_diving_sig(state,1000)
                orders: list[Order] = []
                if last_sign==1:
                    # max long position when the last sign remains positive
                    if legal_buy_vol > 0 and self.best_ask_volume[product] != 0:
                        best_ask = self.best_ask[product]
                        best_ask_volume = self.best_ask_volume[product]
                        orders.append(Order(product, best_ask, int(
                            min(-best_ask_volume, legal_buy_vol))))
                        legal_sell_vol = legal_sell_vol - \
                            int(min(-best_ask_volume, legal_buy_vol))
                        legal_buy_vol = legal_buy_vol - \
                            int(min(-best_ask_volume, legal_buy_vol))
                    result[product] = orders

                elif last_sign == -1:
                    # max short position when the last sign remains negative
                    if legal_sell_vol < 0 and self.best_bid_volume[product] != 0:
                        best_bid = self.best_bid[product]
                        best_bid_volume = self.best_bid_volume[product]
                        orders.append(Order(product, best_bid, int(
                            max(-best_bid_volume, legal_sell_vol))))
                        legal_buy_vol = legal_buy_vol - \
                            int(max(-best_bid_volume, legal_sell_vol))
                        legal_sell_vol = legal_sell_vol - \
                            int(max(-best_bid_volume, legal_sell_vol))
                    result[product] = orders

                else:
                    # last signal = 0, 我们的directional 信号已经过期了，需要清仓
                    if self.current_position[product] < 0:
                        best_ask = self.best_ask[product]
                        best_ask_volume = self.best_ask_volume[product]
                        orders.append(Order(product, best_ask, int(
                            min(-best_ask_volume, -state.position[product]))))
                    elif self.current_position[product] > 0:
                        best_bid = self.best_bid[product]
                        best_bid_volume = self.best_bid_volume[product]
                        orders.append(Order(product, best_bid, int(
                            max(-best_bid_volume, -state.position[product]))))
                    result[product] = orders
            elif product == 'PEARLS' or product == 'BANANAS':
                orders: list[Order] = []
                # 双边都有挂单
                if self.best_ask_volume[product] != 0 and self.best_bid_volume[product] != 0:
                    best_ask = self.best_ask[product]
                    best_ask_volume = self.best_ask_volume[product]
                    best_bid = self.best_bid[product]
                    best_bid_volume = self.best_bid_volume[product]
                    # Check if the lowest ask (sell order) is lower than the above defined fair value
                    # taker strategy
                    # TODO：两个 buy order 可能同时发单，两个 sell order 也可能同时发单

                    if best_ask <= acceptable_price:  # 卖得低，take the sell order on the orderbook as much as possible
                        orders.append(Order(product, best_ask, int(
                            min(-best_ask_volume, legal_buy_vol))))

                        # 因为我们买了，所以我们可以买的量减少，我们可以卖的量增加
                        
                        legal_sell_vol = legal_sell_vol - \
                            int(min(-best_ask_volume, legal_buy_vol))
                        legal_buy_vol = legal_buy_vol - \
                            int(min(-best_ask_volume, legal_buy_vol))

                    if best_bid >= acceptable_price:  # 买得贵，take the buy order on the orderbook as much as possible
                        orders.append(Order(product, best_bid, int(
                            max(-best_bid_volume, legal_sell_vol))))
                        # 因为我们卖了，所以我们可以卖的量减少，我们可以买的量增加
                        legal_buy_vol = legal_buy_vol - \
                            int(max(-best_bid_volume, legal_sell_vol))
                        legal_sell_vol = legal_sell_vol - \
                            int(max(-best_bid_volume, legal_sell_vol))

                    # maker strategy
                    if best_ask - 1 >= acceptable_price:  # orderbook最优卖单足够贵，我们可以以-1的价格仍然卖出获利
                        # make the market by placing a sell order at a price of 1 below the best ask
                        orders.append(
                            Order(product, best_ask - 1, legal_sell_vol))
                    if best_bid + 1 <= acceptable_price:
                        # make the market by placing a buy order at a price of 1 above the best bid
                        orders.append(
                            Order(product, best_bid + 1, legal_buy_vol))
                    result[product]=orders

                # 只有买单
                elif self.best_ask_volume[product] == 0 and self.best_bid_volume[product] != 0:
                    best_bid = self.best_bid[product]
                    best_bid_volume = self.best_bid_volume[product]
                    if best_bid >= acceptable_price:
                        orders.append(Order(product, best_bid, int(
                            max(-best_bid_volume, legal_sell_vol))))
                    result[product]=orders

                # 只有卖单
                elif self.best_ask_volume[product] != 0 and self.best_bid_volume[product] == 0:
                    best_ask = self.best_ask[product]
                    best_ask_volume = self.best_ask_volume[product]
                    if best_ask <= acceptable_price:
                        orders.append(Order(product, best_ask, int(
                            min(-best_ask_volume, legal_buy_vol))))
                    result[product]=orders

        c_order,pc_order = self.pair_trading_1(state)
        result['PINA_COLADAS'] = pc_order
        result['COCONUTS'] = c_order

        P_orders, B_orders, D_orders, U_orders = self.pair_trading_2(state)
        result['PICNIC_BASKET'] = P_orders
        result['BAGUETTE'] = B_orders
        result['DIP'] = D_orders
        result['UKULELE'] = U_orders

        logger.flush(state, result)
        return result