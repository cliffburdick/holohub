# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import asyncio
from typing import Dict
import nats
from functools import partial
from queue import Empty, Queue
from loguru import logger
import time
from attrs import define
from typing import Dict
from common_msg import message_type, ext_msg
from nats.errors import ConnectionClosedError, TimeoutError, NoServersError


async def async_sub_handler(msg: nats.aio.msg.Msg, q: Dict[str,Queue]):
    """Handle incoming messages

    Args:
        msg (NATS message): NATS message type
        q (Queue): Receive queue to push message to
    """
    #logger.debug(f"Message arrived on NATS queue with subject {msg.subject}")
    if msg.subject in q:
        q[msg.subject].put(msg.data, block=False)


@define
class nats_async():
    """Asynchronous handler of ingress and egress NATS messages. Incoming messages
       are required to be in dictionary format and specify a type and a payload.
    """
    host: str     # NATS host    
    rxq: Dict[str, Queue]    # Queue to pass messages on
    txq: Queue    # Queue to send out NATS
    # NATS subscriptions
    sub: Dict[str, nats.aio.subscription.Subscription] = {}
    nc: None = None     # NATS connection
    use_js: bool = False
    js: nats.js.JetStreamContext = None

    def start_async_loop(self):
        """Starts ths asyncio loop
        """
        logger.info("Starting event loop")
        try:
            asyncio.run(self.run())
        except KeyboardInterrupt:
            pass

    async def run(self):
        """Main event loop to do async message processing
        """
        logger.info("Connecting to NATS")
        await self.connect()

        logger.info("Finished connecting")
        # Main processing loop
        while True:
            try:
                el = self.txq.get_nowait()
                if not hasattr(el, 'type'):
                    logger.error("Failed to find message type")
                    continue

                logger.debug(f"Got message type {el.type}")
                # Check internal message type
                if el.type == message_type.EXTERNAL:
                    await self.publish(el)
                elif el.type == message_type.SUBSCRIBE:
                    await self.subscribe(el.subject)
                elif el.type == message_type.SHUTDOWN:
                    await self.close()
                else:
                    logger.error(
                        f"Invalid message type sent to NATS queue: {el.type} {message_type.SUBSCRIBE}")
            except Empty:   
                await asyncio.sleep(0.1)


    async def connect(self):
        """Connect to a NATS JetStream host

        Args:
            host (str): Host to connect to, including port
        """
        try:
            self.nc = await nats.connect(servers=self.host)
            if self.use_js:
                self.js = self.nc.jetstream()
        except ConnectionRefusedError as e:
            logger.error(f"Cannot connect to NATS: {e}")

    async def subscribe(self, subject: str):
        """Subscript to a NATS subject

        Args:
            subject (str): Subject to subscribe to. Can contain wildcards
        """
        logger.info(f"Subscribing to subject {subject}")
        handler = partial(async_sub_handler, q=self.rxq)

        try:
            if self.use_js:
                self.sub['subject'] = await self.js.subscribe(subject, cb=handler)
            else:
                self.sub['subject'] = await self.nc.subscribe(subject, cb=handler)
        except TypeError:
            logger.error(f"Cannot subscribe to {subject} since stream doesn't exist")


    async def publish(self, msg: ext_msg):
        """Publish a message to NATS

        Args:
            subject (str): Subject to publish on
            encoding (encoding_type): Message encoding type
            msg (bytes): Payload
        """
        try:
            if self.use_js:
                await self.js.publish(msg.subject, msg.payload)
            else:
                await self.nc.publish(msg.subject, msg.payload)
        except nats.errors.ConnectionClosedError:
            logger.error("Failed to send message since NATS connect was closed")                

    async def close(self):
        """Close connection to NATS
        """
        logger.info('Closing NATS connection')
        try:
            await self.nc.close()
        except nats.errors.ConnectionClosedError:
            pass
