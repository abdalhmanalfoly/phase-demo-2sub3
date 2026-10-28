"""Injectable wall clock (epoch milliseconds) so health/offline logic is testable without sleeping."""
import time


class SystemClock:
    def now_ms(self) -> int:
        return int(time.time() * 1000)
