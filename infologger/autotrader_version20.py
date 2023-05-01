# Copyright 2021 Optiver Asia Pacific Pty. Ltd.
#
# This file is part of Ready Trader Go.
#
#     Ready Trader Go is free software: you can redistribute it and/or
#     modify it under the terms of the GNU Affero General Public License
#     as published by the Free Software Foundation, either version 3 of
#     the License, or (at your option) any later version.
#
#     Ready Trader Go is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU Affero General Public License for more details.
#
#     You should have received a copy of the GNU Affero General Public
#     License along with Ready Trader Go.  If not, see
#     <https://www.gnu.org/licenses/>.
import asyncio
import itertools
import time
import math
import numpy as np

from typing import List

from ready_trader_go import BaseAutoTrader, Instrument, Lifespan, MAXIMUM_ASK, MINIMUM_BID, Side

# version1.0 copy

"""
conda activate OptiverSys
cd /Users/absolutex/Library/CloudStorage/OneDrive-个人/market/TradingSys/Pyready10L/
python3.11 rtg.py run autotrader.py
"""

### hyper-parameter
LOT_SIZE = 80
SIGMOID_C = math.ceil(math.log(LOT_SIZE-1)/-0.7)
POSITION_LIMIT = 100
TICK_SIZE_IN_CENTS = 100
MIN_BID_NEAREST_TICK = (MINIMUM_BID + TICK_SIZE_IN_CENTS) // TICK_SIZE_IN_CENTS * TICK_SIZE_IN_CENTS
MAX_ASK_NEAREST_TICK = MAXIMUM_ASK // TICK_SIZE_IN_CENTS * TICK_SIZE_IN_CENTS
TIME_NUMBER = 5  # How many periods we look back

OrderDistanceFromBest = 3 # 默认下单的档位，1则在ask1/bid1下单，2则在ask2/bid2下单
ClearPositionPercentage = 0.5 # 超过这个数字乘以最高仓位，我们开始着手准备清仓
trade_memory_num = 40 # 我们记录的交易次数上限

"""
MINIMUM_BID = 1
MAXIMUM_ASK = 2147483647
MinBidNearestBid = 0   always 0
MaxAskNearestTick = 2147483647 // TickSizeInCents**2
"""

"""
/*----- Readme -----*/
Author: AbsoluteX
代码设计请遵循以下原理 目的是确保我们的抢单顺利
1. 优先执行下单等命令
2. 再执行一些记录类 更新类代码
该代码已经更新 将所有的 logger 语句放在了函数最后 且永远在最后
"""

class AutoTrader(BaseAutoTrader):
    """Example Auto-trader.
    When it starts this auto-trader places ten-lot bid and ask orders at the
    current best-bid and best-ask prices respectively. Thereafter, if it has
    a long position (it has bought more lots than it has sold) it reduces its
    bid and ask prices. Conversely, if it has a short position (it has sold
    more lots than it has bought) then it increases its bid and ask prices.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop, team_name: str, secret: str):
        """Initialise a new instance of the AutoTrader class."""
        super().__init__(loop, team_name, secret)
        self.order_ids = itertools.count(1)
        self.bids = set()
        self.asks = set()
        self.ask_id = self.ask_price = self.bid_id = self.bid_price = self.position = 0
        self.bid_num, self.ask_num = 0, 0
     
        """/*----- self-defined parameters -----*/"""
        ### author: AbsoluteX
        #中间变量可以忽略, Intermediate variables, can ignore
        InstrumentOrderbook = {'seq':None,'bidprc':[],'bidqty':[],'askprc':[],'askqty':[]} 
        self.orderbook = {Instrument.FUTURE:InstrumentOrderbook, Instrument.ETF:InstrumentOrderbook}
        # 是否开启logger的指令
        self.if_logger = False #False代表我们关闭logger

        ### author: JL
        self.last_order = dict()
        # 请使用策略命名的全称指代策略
        ### useless ignore
        self.recv_time = 0

    """
    /*----- utils func -----*/
    some utils func
    """

    def sigmoid(self, x):     # Author: Yip
        """input position, return its sigmoid result

        更新: 加入self.penalty,用于趋势交易更改仓位
            1. self.penalty<0, 说明做多区间, 同等仓位下会更多bid单更少ask单
            2. self.penalty>0, 说明做空区间, 同等仓位下更多ask单更少bid单
        """
        return 1/(1+math.exp(-SIGMOID_C*(x)))

    def on_error_message(self, client_order_id: int, error_message: bytes):
        """Called when the exchange detects an error.
        
        If the error pertains to a particular order, then the client_order_id
        will identify that order, otherwise the client_order_id will be zero.
        如果错误与特定订单有关，则 client_order_id 将标识该订单，否则 client_order_id 将为零。
        """
        if client_order_id != 0 and (client_order_id in self.bids or client_order_id in self.asks):
            self.on_order_status_message(client_order_id, 0, 0, 0)

        # put logger in the last line for time optimization
        if self.if_logger: self.logger.warning("error with order %d: %s",
                                               client_order_id, error_message.decode())

    def on_hedge_filled_message(self, client_order_id: int, price: int, volume: int):
        """Called when one of your hedge orders is filled.
        The price is the average price at which the order was (partially) filled,
        which may be better than the order's limit price. The volume is
        the number of lots filled at that price.
        当您的一个对冲订单被执行时调用。价格是订单（部分）成交的平均价格，可能优于订单的限价。 交易量是以该价格成交的手数
        """
        if self.if_logger: self.logger.info("received hedge filled for order %d with average price %d and volume %d",
                                            client_order_id, price, volume)

    def update_order_book(self, instrument: int, sequence_number: int, ask_prices: List[int],
                                     ask_volumes: List[int], bid_prices: List[int], bid_volumes: List[int]):
        """Author: AbsoluteX
        Update orderbook data, when orderbook data arrive, Called by func on_order_book_update_message .
        当新的订单数据到达 更新订单数据 无法自动调用
        同时更新我们的预测订单薄数据 防止预测数据于真实数据产生过大的gap
        """
        self.orderbook[instrument] = {'seq':sequence_number,'bidprc':bid_prices,'bidqty':bid_volumes,'askprc':ask_prices,'askqty':ask_volumes}        
    
    def get_order_num(self):
        """
        Author: Yip
        Adjust order number based on current position
        """
        self.bid_num = int(self.sigmoid(self.position/(2*POSITION_LIMIT))*LOT_SIZE)
        self.ask_num = int((1-self.sigmoid(self.position/(2*POSITION_LIMIT)))*LOT_SIZE)
        
        if self.position >= 0:
            self.ask_num = abs(self.position)+40
        if self.position <= 0:
            self.bid_num = abs(self.position)+40
        
        # 防止爆仓
        if self.position + self.bid_num >= POSITION_LIMIT:
            self.bid_num = POSITION_LIMIT - self.position - 1
        if self.position - self.ask_num <= -POSITION_LIMIT:
            self.ask_num = POSITION_LIMIT + self.position - 1

    #############################################################################################
    ################################## ----Strategy fields---- ##################################
    #############################################################################################
    """                 /*----- please write all your strategies here -----*/                 """

    def ETF_Future_spread_strategy(self, instrument: int, sequence_number: int, ask_prices: List[int],
                                     ask_volumes: List[int], bid_prices: List[int], bid_volumes: List[int]):
        """ Author: Yip AbsoluteX
        我们的 Futures-ETF 配对交易策略主体
        """
        # 策略初始化
        if 'ETF_Future_spread_strategy' not in self.last_order: 
            self.last_order['ETF_Future_spread_strategy'] = {"ask_id":0,"ask_price":0,"bid_id":0,"bid_price":0}

        # Initialize, 如果最终价格等于0我们不会下单
        new_ask_price = 0
        new_bid_price = 0
        if self.orderbook[Instrument.FUTURE]['bidprc'][0] != 0:
            new_bid_price = self.orderbook[Instrument.FUTURE]['bidprc'][OrderDistanceFromBest - 1]
        if self.orderbook[Instrument.FUTURE]['askprc'][0] != 0:
            new_ask_price = self.orderbook[Instrument.FUTURE]['askprc'][OrderDistanceFromBest - 1]

        return new_ask_price, new_bid_price
    
    ################################## Strategy fields ends here ##################################

    def on_order_book_update_message(self, instrument: int, sequence_number: int, ask_prices: List[int],
                                     ask_volumes: List[int], bid_prices: List[int], bid_volumes: List[int]):
        """Called periodically to report the status of an order book.
        The sequence number can be used to detect missed or out-of-order
        messages. The five best available ask (i.e. sell) and bid (i.e. buy)
        prices are reported along with the volume available at each of those
        price levels.
        定期调用报告订单薄的状态
        """
        self.recv_time = time.time()
        self.update_order_book(instrument, sequence_number, ask_prices, ask_volumes, bid_prices, bid_volumes)
        
        ### strategy combo under here
        if instrument == Instrument.FUTURE:
            pass
        if instrument == Instrument.ETF:
            # 防止数据不出现导致报错
            if (self.orderbook[Instrument.ETF]['bidprc']==[])or(self.orderbook[Instrument.ETF]['askprc']==[]): return
            self.block = False      # 初始化
            self.get_order_num()  # Update the order number we will put, according to current position
            ask_prc2, bid_prc2 = self.ETF_Future_spread_strategy(instrument, sequence_number, ask_prices, ask_volumes, bid_prices, bid_volumes)
            self.send_order('ETF_Future_spread_strategy',ask_prc2, bid_prc2, self.ask_num, self.bid_num)

        if self.if_logger: self.logger.info("received order book for instrument %d with sequence number %d", 
                                            instrument,sequence_number)

    def send_order(self,strategy, new_ask_price, new_bid_price, ask_num, bid_num):
        """
        Author: JL
        Descript: 弃用self.ask_id,self.ask_price,self.bid_id,self.bid_price, 启用self.last_order字典 记录两个策略的下单情况
        为两个策略各自开了一个字典分别记录如上四个变量
        """
        # 先取消ask&bid单
        ### 取消前会检查下单价格是不是和存在单价格一样
        if self.last_order[strategy]['bid_id'] != 0 and new_bid_price not in (self.last_order[strategy]['bid_price'], 0):
            self.send_cancel_order(self.last_order[strategy]['bid_id'])
            self.last_order[strategy]['bid_id'] = 0
            
        if self.last_order[strategy]['ask_id'] != 0 and new_ask_price not in (self.last_order[strategy]['ask_price'], 0):
            self.send_cancel_order(self.last_order[strategy]['ask_id'])
            self.last_order[strategy]['ask_id'] = 0
        
        if self.last_order[strategy]['bid_id'] == 0 and new_bid_price != 0 and self.position < POSITION_LIMIT:
            self.last_order[strategy]['bid_id'] = next(self.order_ids)
            self.last_order[strategy]['bid_price'] = new_bid_price
            self.send_insert_order(self.last_order[strategy]['bid_id'], Side.BUY, new_bid_price, bid_num, Lifespan.GOOD_FOR_DAY)
            self.bids.add(self.last_order[strategy]['bid_id'])
        
        if self.last_order[strategy]['ask_id'] == 0 and new_ask_price != 0 and self.position > -POSITION_LIMIT:
            self.last_order[strategy]['ask_id'] = next(self.order_ids)
            self.last_order[strategy]['ask_price'] = new_ask_price
            self.send_insert_order(self.last_order[strategy]['ask_id'], Side.SELL, new_ask_price, ask_num, Lifespan.GOOD_FOR_DAY)
            self.asks.add(self.last_order[strategy]['ask_id'])       

    def on_order_filled_message(self, client_order_id: int, price: int, volume: int):
        """Called when one of your orders is filled, partially or fully.
        The price is the price at which the order was (partially) filled,
        which may be better than the order's limit price. The volume is
        the number of lots filled at that price.
        当您的一个订单被部分或全部成交时调用。
        """
        if client_order_id in self.bids:
            self.position += volume
            self.send_hedge_order(next(self.order_ids), Side.ASK, MIN_BID_NEAREST_TICK, volume)
        elif client_order_id in self.asks:
            self.position -= volume
            self.send_hedge_order(next(self.order_ids), Side.BID, MAX_ASK_NEAREST_TICK, volume)
        
        # put logger in the last line for time optimization
        if self.if_logger: self.logger.info("received order filled for order %d with price %d and volume %d",
                                            client_order_id, price, volume)
    
    def on_order_status_message(self, client_order_id: int, fill_volume: int, remaining_volume: int,
                                fees: int):
        """Called when the status of one of your orders changes.
        The fill_volume is the number of lots already traded, remaining_volume
        is the number of lots yet to be traded and fees is the total fees for
        this order. Remember that you pay fees for being a market taker, but
        you receive fees for being a market maker, so fees can be negative.
        If an order is cancelled its remaining volume will be zero.
        当您的一个订单的状态发生变化时调用。
        """
        if remaining_volume == 0:
            if client_order_id == self.bid_id:
                self.bid_id = 0
            elif client_order_id == self.ask_id:
                self.ask_id = 0

            # It could be either a bid or an ask
            self.bids.discard(client_order_id)
            self.asks.discard(client_order_id)
        
        # put logger in the last line for time optimization
        if self.if_logger: self.logger.info("received order status for order %d with fill volume %d remaining %d and fees %d",
                                            client_order_id, fill_volume, remaining_volume, fees)
  
    def on_trade_ticks_message(self, instrument: int, sequence_number: int, ask_prices: List[int],
                               ask_volumes: List[int], bid_prices: List[int], bid_volumes: List[int]):
        """Called periodically when there is trading activity on the market.
        The five best ask (i.e. sell) and bid (i.e. buy) prices at which there
        has been trading activity are reported along with the aggregated volume
        traded at each of those price levels.
        If there are less than five prices on a side, then zeros will appear at
        the end of both the prices and volumes arrays.
        当市场上有交易活动时定期调用。
        """
        

        # put logger in the last line for time optimization
        if self.if_logger: self.logger.info("received trade ticks for instrument %d with sequence number %d",
                                            instrument, sequence_number)