
import json
from datamodel import Order, ProsperityEncoder, Symbol, TradingState, OrderDepth
from typing import Any, Dict, List
import numpy as np

class Logger:
    def __init__(self) -> None:
        self.logs = ""

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]]) -> None:
        print(json.dumps({
            "state": state,
            "orders": orders,
            "logs": self.logs,
        }, cls=ProsperityEncoder, separators=(",", ":"), sort_keys=True))

        self.logs = ""

logger = Logger()

class Trader:
    # define data members
    def __init__(self):
        self.position_limit = {"PEARLS": 20, "BANANAS": 20,'COCONUTS':600,'PINA_COLADAS':300,'BERRIES':250,'DIVING_GEAR':50,'DOLPHIN_SIGHTINGS':0}
        self.last_mid_price = {'PEARLS': 10000, 'BANANAS': 4900,'COCONUTS':8000,'PINA_COLADAS':15000,'BERRIES':4000,'DIVING_GEAR':10000,'DOLPHIN_SIGHTINGS':0}
        self.acceptable_price = {'PEARLS': 10000, 'BANANAS': 5000,'COCONUTS': 8000,'PINA_COLADAS': 15000,'BERRIES':4000,'DIVING_GEAR':10000,'DOLPHIN_SIGHTINGS':0}
        self.legal_buy_vol = {'PEARLS': 20, 'BANANAS': 20,'COCONUTS': 600,'PINA_COLADAS': 300,'BERRIES':250,'DIVING_GEAR':50,'DOLPHIN_SIGHTINGS':0}
        self.legal_sell_vol = {'PEARLS': -20, 'BANANAS': -20,'COCONUTS': -600,'PINA_COLADAS': -300,'BERRIES':-250,'DIVING_GEAR':-50,'DOLPHIN_SIGHTINGS':0}
        self.current_position = {'PEARLS': 0, 'BANANAS': 0,'COCONUTS': 0,'PINA_COLADAS': 0,'BERRIES':0,'DIVING_GEAR':0,'DOLPHIN_SIGHTINGS':0}
        self.last_sign = 0
        self.signal_expire = 0
        # store the orderbook depth 1 for each product
        self.best_bid = {'PEARLS': 0, 'BANANAS': 0,'COCONUTS': 0,'PINA_COLADAS': 0,'BERRIES':0,'DIVING_GEAR':0,'DOLPHIN_SIGHTINGS':0}
        self.best_ask = {'PEARLS': 0, 'BANANAS': 0,'COCONUTS': 0,'PINA_COLADAS': 0,'BERRIES':0,'DIVING_GEAR':0,'DOLPHIN_SIGHTINGS':0}
        self.best_bid_volume = {'PEARLS': 0, 'BANANAS': 0,'COCONUTS': 0,'PINA_COLADAS': 0,'BERRIES':0,'DIVING_GEAR':0,'DOLPHIN_SIGHTINGS':0}
        self.best_ask_volume = {'PEARLS': 0, 'BANANAS': 0,'COCONUTS': 0,'PINA_COLADAS': 0,'BERRIES':0,'DIVING_GEAR':0,'DOLPHIN_SIGHTINGS':0}
        self.last_best_bid = {'PEARLS': 0, 'BANANAS': 0,'COCONUTS': 0,'PINA_COLADAS': 0,'BERRIES':0,'DIVING_GEAR':0,'DOLPHIN_SIGHTINGS':0}
        self.last_best_ask = {'PEARLS': 0, 'BANANAS': 0,'COCONUTS': 0,'PINA_COLADAS': 0,'BERRIES':0,'DIVING_GEAR':0,'DOLPHIN_SIGHTINGS':0}

    def market_status(self, state: TradingState, product) -> int:
        order_depth: OrderDepth = state.order_depths[product]
        # Diving gear 的volume imbalance 预测一般（corr 0.1），其他三个品种还不错，banana*1, pearl*2, berries *0.5
        if product == 'BERRIES':
            if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
                best_ask_volume = order_depth.sell_orders[min(order_depth.sell_orders.keys())]
                best_bid_volume = order_depth.buy_orders[max(order_depth.buy_orders.keys())]
                if -best_ask_volume > 30 and best_bid_volume <= 30:
                    return -1
                elif -best_ask_volume <= 30 and best_bid_volume > 30:
                    return 1
                else:
                    return 0
        elif product == 'PEARLS' or product == 'BANANAS':
            if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
                best_ask_volume = order_depth.sell_orders[min(order_depth.sell_orders.keys())]
                best_bid_volume = order_depth.buy_orders[max(order_depth.buy_orders.keys())]
                if -best_ask_volume > 15 and best_bid_volume <= 15:
                    return -1
                elif -best_ask_volume <= 15 and best_bid_volume > 15:
                    return 1
                else:
                    return 0
        else:
            return 0


    
    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        
        # Initialize the method output dict as an empty dict
        result = {}
        # Iterate over all the keys (the available products) contained in the order depths

        # for each product, we read its orderbook at first
        # to get:
        # 1. the fair price(weighted mid price)
        # 2. the legal buy volume and legal sell volume

        # generate signal for the diving gear
        dolphin_sightings = state.observations['DOLPHIN_SIGHTINGS']
        if self.last_mid_price['DOLPHIN_SIGHTINGS'] == 0:
            self.last_mid_price['DOLPHIN_SIGHTINGS'] = dolphin_sightings
        dolphin_sightings_diff = dolphin_sightings-self.last_mid_price['DOLPHIN_SIGHTINGS']
        self.last_mid_price['DOLPHIN_SIGHTINGS'] = dolphin_sightings 
        
        instant_signal = 1 if dolphin_sightings_diff>5 else -1 if dolphin_sightings_diff< -5 else 0
        

        # we save the dg_signal for 1000 seconds

        if instant_signal ==1:
            self.last_sign = 1
            self.signal_expire = state.timestamp + 1000*100
        elif instant_signal == -1:
            self.last_sign = -1
            self.signal_expire = state.timestamp + 1000*100
        else:
            if state.timestamp > self.signal_expire:
                self.last_sign = 0
            else:
                # 不更改last sign 继续记忆上一个instant signal
                pass


        for product in state.order_depths.keys():
            
            if product in state.position.keys():
                current_position = state.position[product]
            else:
                current_position = 0
            legal_buy_vol = self.position_limit[product] - current_position # -pos_limit <=cur<=pos_limit, so the legal buy vol is always non-negative
            legal_sell_vol = -self.position_limit[product] - current_position # -pos_limit <=cur<=pos_limit, so the legal sell vol is always non-positive

            # Retrieve the Order Depth containing all the market BUY and SELL orders for PEARLS
            order_depth: OrderDepth = state.order_depths[product]
            # Calculate the fair price based on the order depth
            if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
                best_ask = min(order_depth.sell_orders.keys())
                best_ask_volume = order_depth.sell_orders[best_ask]
                best_bid = max(order_depth.buy_orders.keys())
                best_bid_volume = order_depth.buy_orders[best_bid]
                avg_buy_price = sum([order_depth.buy_orders[i] * i for i in order_depth.buy_orders.keys()])/sum([order_depth.buy_orders[i] for i in order_depth.buy_orders.keys()])
                avg_sell_price = sum([-order_depth.sell_orders[i] * i for i in order_depth.sell_orders.keys()])/sum([-order_depth.sell_orders[i] for i in order_depth.sell_orders.keys()])
                # mid_price = (avg_sell_price + avg_buy_price)/2 weighted bid weighted ask/2
                buy_order_volume = sum([order_depth.buy_orders[i] for i in order_depth.buy_orders.keys()])
                sell_order_volume = sum([-order_depth.sell_orders[i] for i in order_depth.sell_orders.keys()])
                mid_price = (avg_sell_price * buy_order_volume + avg_buy_price * sell_order_volume)/(buy_order_volume + sell_order_volume)

                
            elif len(order_depth.sell_orders) == 0 and len(order_depth.buy_orders) != 0: # 只有买单
                best_ask = 0
                best_ask_volume = 0
                best_bid = max(order_depth.buy_orders.keys())
                best_bid_volume = order_depth.buy_orders[best_bid]
                mid_price = self.last_mid_price[product]
            elif len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) == 0: # 只有卖单
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
            self.last_mid_price[product] = mid_price
            self.legal_buy_vol[product] = legal_buy_vol
            self.legal_sell_vol[product] = legal_sell_vol
            self.current_position[product] = current_position
            
            

        # for each product, we calculate the acceptable price,
        for product in state.order_depths.keys():
                # the relationship between coconut and pina colada linear with intercept, 2*coconut = pina colada+1000
            
            if product == 'PEARLS':
                market_status = self.market_status(state, product)
                self.acceptable_price[product] = self.last_mid_price[product]+2*market_status
            elif product == 'BANANAS':
                market_status = self.market_status(state, product)
                self.acceptable_price[product] = self.last_mid_price[product]+1*market_status
            elif product == 'BERRIES':
                market_status = self.market_status(state, product)
                self.acceptable_price[product] = self.last_mid_price[product] + 0.5 * market_status

            elif product == 'COCONUTS':
                self.acceptable_price[product] = (self.last_mid_price[product]+((self.last_mid_price['PINA_COLADAS']+1000)/2))/2
            elif product == 'PINA_COLADAS':
                self.acceptable_price[product] = (self.last_mid_price[product] +(self.last_mid_price['COCONUTS']*2-1000))/2
            elif product == 'DIVING_GEAR':
                self.acceptable_price[product] = self.last_mid_price[product]




            # Initialize the list of Orders to be sent as an empty list
            orders: list[Order] = []


            # get acceptable price, legal buy volume and legal sell volume from the init
            
            acceptable_price = self.acceptable_price[product]
            legal_buy_vol = self.legal_buy_vol[product]
            legal_sell_vol = self.legal_sell_vol[product]
            if product == 'BERRIES':
                if state.timestamp < 250000 or state.timestamp >= 780000:
                    if self.best_ask_volume[product] !=0 and self.best_bid_volume[product] !=0: # 双边都有挂单
                        best_ask = self.best_ask[product]
                        best_ask_volume = self.best_ask_volume[product]
                        best_bid = self.best_bid[product]
                        best_bid_volume = self.best_bid_volume[product]
                        # Check if the lowest ask (sell order) is lower than the above defined fair value
                        # taker strategy
                        # TODO：两个 buy order 可能同时发单，两个 sell order 也可能同时发单
                        if best_ask <= acceptable_price:# 卖得低，take the sell order on the orderbook as much as possible
                            orders.append(Order(product, best_ask, int(min(-best_ask_volume,legal_buy_vol))))

                            # 因为我们买了，所以我们可以买的量减少，我们可以卖的量增加
                            # TODO: 先更新 sell, 如果先更新buy,那更新sell的时候数值已经变了
                            legal_sell_vol = legal_sell_vol - int(min(-best_ask_volume,legal_buy_vol))
                            legal_buy_vol = legal_buy_vol - int(min(-best_ask_volume,legal_buy_vol))
                        if best_bid >= acceptable_price: # 买得贵，take the buy order on the orderbook as much as possible
                            orders.append(Order(product, best_bid, int(max(-best_bid_volume,legal_sell_vol))))
                            # 因为我们卖了，所以我们可以卖的量减少，我们可以买的量增加
                            legal_buy_vol = legal_buy_vol - int(max(-best_bid_volume,legal_sell_vol))
                            legal_sell_vol = legal_sell_vol - int(max(-best_bid_volume,legal_sell_vol)) 
                            
                        
                        # maker strategy
                        if best_ask - 1 >= acceptable_price:# orderbook最优卖单足够贵，我们可以以-1的价格仍然卖出获利
                            # make the market by placing a sell order at a price of 1 below the best ask
                            orders.append(Order(product, best_ask - 1, legal_sell_vol))
                        if best_bid + 1 <= acceptable_price:
                            # make the market by placing a buy order at a price of 1 above the best bid
                            orders.append(Order(product, best_bid + 1, legal_buy_vol))

                    elif self.best_ask_volume[product] ==0 and self.best_bid_volume[product] !=0: # 只有买单
                        best_bid = self.best_bid[product]
                        best_bid_volume = self.best_bid_volume[product]
                        if best_bid >= acceptable_price:
                            orders.append(Order(product, best_bid, int(max(-best_bid_volume,legal_sell_vol))))

                    elif self.best_ask_volume[product] !=0 and self.best_bid_volume[product] ==0: # 只有卖单
                        best_ask = self.best_ask[product]
                        best_ask_volume = self.best_ask_volume[product]
                        if best_ask <= acceptable_price:
                            orders.append(Order(product, best_ask, int(min(-best_ask_volume,legal_buy_vol))))
                    # Add all the above the orders to the result dict
                    result[product] = orders
                elif state.timestamp >= 250000 and state.timestamp < 280000:
                    if legal_buy_vol > 0 and self.best_ask_volume[product] !=0:
                        best_ask = self.best_ask[product]
                        best_ask_volume = self.best_ask_volume[product]
                        orders.append(Order(product, best_ask, int(min(-best_ask_volume,legal_buy_vol))))
                        legal_sell_vol = legal_sell_vol - int(min(-best_ask_volume,legal_buy_vol))
                        legal_buy_vol = legal_buy_vol - int(min(-best_ask_volume,legal_buy_vol))
                    result[product] = orders
                elif state.timestamp >= 500000 and state.timestamp < 530000:
                    if legal_sell_vol < 0 and self.best_bid_volume[product] !=0:
                        best_bid = self.best_bid[product]
                        best_bid_volume = self.best_bid_volume[product]
                        orders.append(Order(product, best_bid, int(max(-best_bid_volume,legal_sell_vol))))
                        legal_buy_vol = legal_buy_vol - int(max(-best_bid_volume,legal_sell_vol))
                        legal_sell_vol = legal_sell_vol - int(max(-best_bid_volume,legal_sell_vol))
                    result[product] = orders
                elif state.timestamp >= 750000 and state.timestamp < 780000:
                    if state.position[product] < 0:
                        best_ask = self.best_ask[product]
                        best_ask_volume = self.best_ask_volume[product]
                        orders.append(Order(product, best_ask, int(min(-best_ask_volume, -state.position[product]))))
                    result[product] = orders
            
            elif product =='DIVING_GEAR':
                if self.last_sign == 1:
                    # max long position until 1000 tick(1000*100 timestamp delta) has passed
                    if legal_buy_vol > 0 and self.best_ask_volume[product] !=0:
                        best_ask = self.best_ask[product]
                        best_ask_volume = self.best_ask_volume[product]
                        orders.append(Order(product, best_ask, int(min(-best_ask_volume,legal_buy_vol))))
                        legal_sell_vol = legal_sell_vol - int(min(-best_ask_volume,legal_buy_vol))
                        legal_buy_vol = legal_buy_vol - int(min(-best_ask_volume,legal_buy_vol))
                    result[product] = orders

                elif self.last_sign == -1:
                    # max short position until 1000 tick(1000*100 timestamp delta) has passed
                    if legal_sell_vol < 0 and self.best_bid_volume[product] !=0:
                        best_bid = self.best_bid[product]
                        best_bid_volume = self.best_bid_volume[product]
                        orders.append(Order(product, best_bid, int(max(-best_bid_volume,legal_sell_vol))))
                        legal_buy_vol = legal_buy_vol - int(max(-best_bid_volume,legal_sell_vol))
                        legal_sell_vol = legal_sell_vol - int(max(-best_bid_volume,legal_sell_vol))
                    result[product] = orders

                else:
                    # last signal = 0, 我们的directional 信号已经过期了，需要清仓
                    if self.current_position[product] < 0:
                        best_ask = self.best_ask[product]
                        best_ask_volume = self.best_ask_volume[product]
                        orders.append(Order(product, best_ask, int(min(-best_ask_volume, -state.position[product]))))
                    elif self.current_position[product] > 0:
                        best_bid = self.best_bid[product]
                        best_bid_volume = self.best_bid_volume[product]
                        orders.append(Order(product, best_bid, int(max(-best_bid_volume, -state.position[product]))))
                    else:
                        pass
                    result[product] = orders

            else:
                if self.best_ask_volume[product] !=0 and self.best_bid_volume[product] !=0: # 双边都有挂单
                    best_ask = self.best_ask[product]
                    best_ask_volume = self.best_ask_volume[product]
                    best_bid = self.best_bid[product]
                    best_bid_volume = self.best_bid_volume[product]
                    # Check if the lowest ask (sell order) is lower than the above defined fair value
                    # taker strategy
                    # TODO：两个 buy order 可能同时发单，两个 sell order 也可能同时发单
                    
                    if best_ask <= acceptable_price:# 卖得低，take the sell order on the orderbook as much as possible
                        orders.append(Order(product, best_ask, int(min(-best_ask_volume,legal_buy_vol))))

                        # 因为我们买了，所以我们可以买的量减少，我们可以卖的量增加
                        # TODO: 先更新 sell, 如果先更新buy,那更新sell的时候数值已经变了
                        legal_sell_vol = legal_sell_vol - int(min(-best_ask_volume,legal_buy_vol))
                        legal_buy_vol = legal_buy_vol - int(min(-best_ask_volume,legal_buy_vol))

                    if best_bid >= acceptable_price: # 买得贵，take the buy order on the orderbook as much as possible
                        orders.append(Order(product, best_bid, int(max(-best_bid_volume,legal_sell_vol))))
                        # 因为我们卖了，所以我们可以卖的量减少，我们可以买的量增加
                        legal_buy_vol = legal_buy_vol - int(max(-best_bid_volume,legal_sell_vol))
                        legal_sell_vol = legal_sell_vol - int(max(-best_bid_volume,legal_sell_vol)) 
                        
                    
                    # maker strategy
                    if best_ask - 1 >= acceptable_price:# orderbook最优卖单足够贵，我们可以以-1的价格仍然卖出获利
                        # make the market by placing a sell order at a price of 1 below the best ask
                        orders.append(Order(product, best_ask - 1, legal_sell_vol))
                    if best_bid + 1 <= acceptable_price:
                        # make the market by placing a buy order at a price of 1 above the best bid
                        orders.append(Order(product, best_bid + 1, legal_buy_vol))

                elif self.best_ask_volume[product] ==0 and self.best_bid_volume[product] !=0: # 只有买单
                    best_bid = self.best_bid[product]
                    best_bid_volume = self.best_bid_volume[product]
                    if best_bid >= acceptable_price:
                        orders.append(Order(product, best_bid, int(max(-best_bid_volume,legal_sell_vol))))

                elif self.best_ask_volume[product] !=0 and self.best_bid_volume[product] ==0: # 只有卖单
                    best_ask = self.best_ask[product]
                    best_ask_volume = self.best_ask_volume[product]
                    if best_ask <= acceptable_price:
                        orders.append(Order(product, best_ask, int(min(-best_ask_volume,legal_buy_vol))))

            # Add all the above the orders to the result dict
            result[product] = orders
            self.last_best_bid[product] = best_bid
            self.last_best_ask[product] = best_ask
        
       


        logger.flush(state, result)   
        return result
