# PEARLS and BANANAS
# fair price = weighte mid price
# C and PC
# C fair price = weighted mid price of C + coef * weighted mid price of PC
# PC fair price = weighted mid price of PC + coef * weighted mid price of C

# import json
from datamodel import Order, ProsperityEncoder, Symbol, TradingState, OrderDepth
from typing import Any, Dict, List

# class Logger:
#     def __init__(self) -> None:
#         self.logs = ""

#     def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
#         self.logs += sep.join(map(str, objects)) + end

#     def flush(self, state: TradingState, orders: dict[Symbol, list[Order]]) -> None:
#         print(json.dumps({
#             "state": state,
#             "orders": orders,
#             "logs": self.logs,
#         }, cls=ProsperityEncoder, separators=(",", ":"), sort_keys=True))

#         self.logs = ""

# logger = Logger()

class Trader:
    # define data members
    def __init__(self):
        self.position_limit = {"PEARLS": 20, "BANANAS": 20,"COCONUTS":600,"PINA_COLADAS":300}
        self.last_mid_price = {'PEARLS': 10000, 'BANANAS': 5000,'COCONUTS': 8000,'PINA_COLADAS': 15000}
        self.acceptable_price = {'PEARLS': 10000, 'BANANAS': 5000,'COCONUTS': 8000,'PINA_COLADAS': 15000}
        self.legal_buy_vol = {'PEARLS': 20, 'BANANAS': 20,'COCONUTS': 600,'PINA_COLADAS': 300}
        self.legal_sell_vol = {'PEARLS': 20, 'BANANAS': 20,'COCONUTS': 600,'PINA_COLADAS': 300}
        # store the orderbook depth 1 for each product
        self.best_bid = {'PEARLS': 0, 'BANANAS': 0,'COCONUTS': 0,'PINA_COLADAS': 0}
        self.best_ask = {'PEARLS': 0, 'BANANAS': 0,'COCONUTS': 0,'PINA_COLADAS': 0}
        self.best_bid_volume = {'PEARLS': 0, 'BANANAS': 0,'COCONUTS': 0,'PINA_COLADAS': 0}
        self.best_ask_volume = {'PEARLS': 0, 'BANANAS': 0,'COCONUTS': 0,'PINA_COLADAS': 0}



    
    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        
        # Initialize the method output dict as an empty dict
        result = {}
        # Iterate over all the keys (the available products) contained in the order depths

        # for each product, we read its orderbook at first
        # to get:
        # 1. the fair price(weighted mid price)
        # 2. the legal buy volume and legal sell volume

        for product in state.order_depths.keys():
            
            # 算仓位信息，和 orderbook 无关
            if product in state.position.keys():
                current_position = state.position[product]
            else:
                current_position = 0
            
            legal_buy_vol = self.position_limit[product] - current_position # -pos_limit <=cur<=pos_limit, so the legal buy vol is always non-negative
            legal_sell_vol = -self.position_limit[product] - current_position # -pos_limit <=cur<=pos_limit, so the legal sell vol is always non-positive
            # legal_buy_vol = min(self.position_limit[product] - current_position,self.position_limit[product])# positive
            # legal_sell_vol = max(-(self.position_limit[product] + current_position),-self.position_limit[product]) # negative

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
                buy_order_volume = sum([order_depth.buy_orders[i] for i in order_depth.buy_orders.keys()])
                sell_order_volume = sum([-order_depth.sell_orders[i] for i in order_depth.sell_orders.keys()])
                mid_price = (avg_sell_price*sell_order_volume + avg_buy_price*buy_order_volume)/(sell_order_volume+buy_order_volume)
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
            

        # for each product, we place the order
        for product in state.order_depths.keys():
                # the relationship between coconut and pina colada linear with intercept, 2*coconut = pina colada+1000

            if product == 'PEARLS' or product == 'BANANAS':
                self.acceptable_price[product] = self.last_mid_price[product]
            elif product == 'COCONUTS':
                self.acceptable_price[product] = (self.last_mid_price[product]+((self.last_mid_price['PINA_COLADAS']+1000)/2))/2
            elif product == 'PINA_COLADAS':
                self.acceptable_price[product] = (self.last_mid_price[product] +(self.last_mid_price['COCONUTS']*2-1000))/2
            else:
                self.acceptable_price[product] = self.last_mid_price[product]

            # Initialize the list of Orders to be sent as an empty list
            orders: list[Order] = []


            # get acceptable price, legal buy volume and legal sell volume from the init
            acceptable_price = self.acceptable_price[product]
            legal_buy_vol = self.legal_buy_vol[product]
            legal_sell_vol = self.legal_sell_vol[product]
            
            # load the orderbook for one more time to place the order
       


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


        # logger.flush(state, result)   
        return result
