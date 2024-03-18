import logging
import sys
import os

local_rank = int(os.environ.get("LOCAL_RANK", -1))

# normal logger
logger = logging.getLogger(__name__)
handler_shared = logging.StreamHandler(sys.stdout)
formatter_shared = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
handler_shared.setFormatter(formatter_shared)
logger.addHandler(handler_shared)

# rank-specific logger
if local_rank != -1:
    logger_rank_specific = logging.getLogger(__name__ + "rank_specific")
    handler_rank_specific = logging.StreamHandler(sys.stdout)
    formatter_rank_specific = logging.Formatter(f'%(asctime)s - %(levelname)s - p{local_rank} - %(name)s - %(message)s')
    handler_rank_specific.setFormatter(formatter_rank_specific)
    logger_rank_specific.addHandler(handler_rank_specific)
else:
    logger_rank_specific = logger

logger_rank_specific.setLevel(logging.DEBUG)

logger.setLevel(logging.DEBUG if local_rank < 1 else logging.WARN)


GetLogger = logger_rank_specific