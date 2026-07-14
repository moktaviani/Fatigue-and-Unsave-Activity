import time

class EventTracker:
    def __init__(self, duration_threshold, event_limit, miss_tolerance=5):
        self.duration_threshold = duration_threshold
        self.event_limit = event_limit
        self.miss_tolerance = miss_tolerance

        self.active_start = None   
        self.miss_count = 0       
        self.counted = False      
        self.event_count = 0      
        self.last_event_time = None
        self.alert_triggered_at = None  
        self.alert_occurrence_count = 0  
        self.just_triggered = False     

    def update(self, detected: bool, now: float = None):
        if now is None:
            now = time.time()

        if detected:
            self.miss_count = 0
            if self.active_start is None:
                self.active_start = now
                self.counted = False

            duration = now - self.active_start
            if duration >= self.duration_threshold and not self.counted:
                self.event_count += 1
                self.counted = True
                self.last_event_time = now
        else:
            if self.active_start is not None:
                self.miss_count += 1
                if self.miss_count > self.miss_tolerance:
                    self.active_start = None
                    self.miss_count = 0
                    self.counted = False

    def current_duration(self, now: float = None):
        if self.active_start is None:
            return 0.0
        if now is None:
            now = time.time()
        return now - self.active_start

    def is_alert(self, now: float = None, display_duration: float = None):
        self.just_triggered = False

        if self.event_count < self.event_limit:
            return False

        if now is None:
            now = time.time()

        if display_duration is None:
            return True

        if self.alert_triggered_at is None:
            self.alert_triggered_at = now
            self.alert_occurrence_count += 1  
            self.just_triggered = True

        if now - self.alert_triggered_at > display_duration:
            self.reset()
            return False

        return True

    def reset_if_idle(self, now: float = None, reset_window: float = 60.0):
        if now is None:
            now = time.time()
        if self.last_event_time and (now - self.last_event_time) > reset_window:
            self.event_count = 0
            self.last_event_time = None

    def reset(self):
        self.active_start = None
        self.miss_count = 0
        self.counted = False
        self.event_count = 0
        self.last_event_time = None
        self.alert_triggered_at = None
