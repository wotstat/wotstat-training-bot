# -*- coding: utf-8 -*-
import BigWorld
import re
import types
import sys
from PlayerEvents import g_playerEvents
from constants import PREBATTLE_TYPE
from adisp import adisp_process, adisp_async
from helpers import dependency
from skeletons.gui.shared.utils import IHangarSpace

VERSION = '{{VERSION}}'
DEBUG_MODE = '{{DEBUG_MODE}}'

# Cooldown time in seconds before accepting a new invite
INVITE_COOLDOWN = 15.0

# Auto-reconnect settings
RECONNECT_INTERVAL = 30.0  # seconds between reconnect attempts


# Minimum similarity threshold (0.0 to 1.0) for fuzzy vehicle matching
# 0.5 means at least 50% of characters must match
VEHICLE_MATCH_THRESHOLD = 0.5


def log(msg):
    print('[MOD_WOTSTAT_TRAINING_BOT] %s' % msg)


def debug(msg):
    if DEBUG_MODE:
        print('[MOD_WOTSTAT_TRAINING_BOT] [DEBUG] %s' % msg)


# Default values
DEFAULT_TARGET_SERVER_NAME = 'RU2'
DEFAULT_ALLOWED_PLAYERS = []
DEFAULT_ALLOWED_PLAYERS_REGEX = []

# Parse arguments at module load time
TARGET_SERVER_NAME = 'RU2'
ALLOWED_PLAYERS = []
ALLOWED_PLAYERS_REGEX = []

# ModsSettings API linkage
MOD_LINKAGE = 'wotstat.training-bot'

MOD_SETTINGS_TEMPLATE = {
    'modDisplayName': 'WotStat Training Bot',
    'settingsVersion': 1,
    'enabled': True,
    'column1': [
        {
            'type': 'Label',
            'text': 'Server Settings',
        },
        {
            'type': 'TextInput',
            'text': 'Target Server Name',
            'width': 200,
            'value': DEFAULT_TARGET_SERVER_NAME,
            'varName': 'targetServerName',
            'tooltip': '{HEADER}Target Server{/HEADER}{BODY}Server name for auto-reconnect (e.g. RU2, RU1){/BODY}',
        },
        {
            'type': 'Empty',
        },
        {
            'type': 'Label',
            'text': 'Allowed Players',
        },
        {
            'type': 'TextInput',
            'text': 'Exact Player Names (comma separated)',
            'width': 350,
            'value': '',
            'varName': 'allowedPlayers',
            'tooltip': '{HEADER}Allowed Players{/HEADER}{BODY}Comma separated list of exact player names allowed to send training invites. Leave empty to accept from everyone.{/BODY}',
        },
    ],
    'column2': [
        {
            'type': 'Label',
            'text': '\xc2\xa0',
        },
        {
            'type': 'TextInput',
            'text': 'Regex Patterns (comma separated)',
            'width': 350,
            'value': '',
            'varName': 'allowedPlayersRegex',
            'tooltip': '{HEADER}Regex Patterns{/HEADER}{BODY}Comma separated list of regex patterns for allowed player names. Leave empty to skip regex matching.{/BODY}',
        },
    ],
}


def _applySettings(settings):
    """Apply settings from ModsSettings API to global variables."""
    global TARGET_SERVER_NAME, ALLOWED_PLAYERS, ALLOWED_PLAYERS_REGEX

    if settings is None:
        return

    # Target server name
    serverName = settings.get('targetServerName', '').strip()
    if serverName:
        TARGET_SERVER_NAME = serverName
    else:
        TARGET_SERVER_NAME = DEFAULT_TARGET_SERVER_NAME

    # Allowed players (comma separated)
    playersStr = settings.get('allowedPlayers', '').strip()
    if playersStr:
        ALLOWED_PLAYERS = [p.strip() for p in playersStr.split(',') if p.strip()]
    else:
        ALLOWED_PLAYERS = list(DEFAULT_ALLOWED_PLAYERS)

    # Allowed players regex (comma separated)
    regexStr = settings.get('allowedPlayersRegex', '').strip()
    if regexStr:
        ALLOWED_PLAYERS_REGEX = [p.strip() for p in regexStr.split(',') if p.strip()]
    else:
        ALLOWED_PLAYERS_REGEX = list(DEFAULT_ALLOWED_PLAYERS_REGEX)

    log('Settings applied: target_server=%s, allowed_players=%s, allowed_players_regex=%s'
        % (TARGET_SERVER_NAME, ALLOWED_PLAYERS, ALLOWED_PLAYERS_REGEX))


def _onModSettingsChanged(linkage, settings):
    """Callback for ModsSettings API when settings change."""
    if linkage != MOD_LINKAGE:
        return
    log('Settings changed via ModsSettings API')
    _applySettings(settings)


def _initModsSettings():
    """Initialize ModsSettings API integration."""
    try:
        from gui.modsSettingsApi import g_modsSettingsApi

        savedSettings = g_modsSettingsApi.getModSettings(MOD_LINKAGE, MOD_SETTINGS_TEMPLATE)
        if savedSettings:
            _applySettings(savedSettings)
            g_modsSettingsApi.registerCallback(MOD_LINKAGE, _onModSettingsChanged)
        else:
            settings = g_modsSettingsApi.setModTemplate(MOD_LINKAGE, MOD_SETTINGS_TEMPLATE, _onModSettingsChanged)
            _applySettings(settings)

        log('ModsSettings API initialized')
    except ImportError:
        log('ModsSettings API not available, using default settings')
    except Exception as e:
        log('Error initializing ModsSettings API: %s' % str(e))


def safeEncode(s):
    """Safely encode unicode string to UTF-8, or return as-is if already bytes."""
    if isinstance(s, types.UnicodeType):
        return s.encode('utf-8')
    return s


def toUnicode(s):
    """Convert string to unicode for proper comparison."""
    if isinstance(s, types.UnicodeType):
        return s
    if isinstance(s, str):
        return s.decode('utf-8', errors='ignore')
    return unicode(s)


def normalizeString(s):
    """Normalize string for fuzzy matching: lowercase, remove special chars, spaces."""
    # Convert to unicode for proper character handling
    s = toUnicode(s)
    s = s.lower()
    # Remove common separators and special characters
    s = re.sub(u'[-_\\s.:\/\\\\]+', u'', s)
    return s


def calculateSimilarity(query, target):
    """
    Calculate similarity between query and target strings.
    Returns a score between 0.0 and 1.0.
    Uses a combination of substring matching and character-level similarity.
    """
    query = normalizeString(query)
    target = normalizeString(target)
    
    if not query or not target:
        return 0.0
    
    # Exact match after normalization
    if query == target:
        return 1.0
    
    # Check if query is a substring of target (high priority)
    if query in target:
        return 0.9 + (0.1 * len(query) / len(target))
    
    # Check if target is a substring of query
    if target in query:
        return 0.8 + (0.1 * len(target) / len(query))
    
    # Longest common subsequence ratio
    lcs = longestCommonSubsequence(query, target)
    lcsRatio = (2.0 * lcs) / (len(query) + len(target))
    
    # Character overlap ratio
    queryChars = set(query)
    targetChars = set(target)
    if queryChars and targetChars:
        overlap = len(queryChars & targetChars)
        charRatio = float(overlap) / max(len(queryChars), len(targetChars))
    else:
        charRatio = 0.0
    
    # Combine ratios with weights
    return 0.7 * lcsRatio + 0.3 * charRatio


def longestCommonSubsequence(s1, s2):
    """Calculate length of longest common subsequence between two strings."""
    m, n = len(s1), len(s2)
    # Use space-optimized approach
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, prev
    
    return prev[n]


def findBestVehicleMatch(query, vehicles):
    """
    Find the best matching vehicle for the given query.
    Returns (vehicle, score) or (None, 0) if no match above threshold.
    """
    bestMatch = None
    bestScore = 0.0
    
    for vehicle in vehicles:
        # Match against user-visible name (localized)
        userNameScore = calculateSimilarity(query, vehicle.userName)
        
        # Match against short user name
        shortNameScore = calculateSimilarity(query, vehicle.shortUserName)
        
        # Match against technical tag (e.g., "ussr:R45_IS-7")
        tagName = vehicle.descriptor.name
        tagScore = calculateSimilarity(query, tagName)
        
        # Also try matching just the vehicle part of tag (after colon)
        if ':' in tagName:
            vehiclePart = tagName.split(':')[1]
            vehiclePartScore = calculateSimilarity(query, vehiclePart)
            tagScore = max(tagScore, vehiclePartScore)
        
        # Take the best score from all match types
        score = max(userNameScore, shortNameScore, tagScore)
        
        if score > bestScore:
            bestScore = score
            bestMatch = vehicle
    
    if bestScore >= VEHICLE_MATCH_THRESHOLD:
        return (bestMatch, bestScore)
    return (None, 0.0)


class AutoReconnectController(object):
    """
    Controller for automatically reconnecting to RU2 server
    when disconnected due to server restart or error.
    """
    
    def __init__(self):
        self._initialized = False
        self._reconnectCallback = None
        self._isReconnecting = False
        self._originalShowDisconnect = None
        self._suppressDisconnectDialog = False
        self._connectionMgr = None
    
    def start(self):
        """Start listening for disconnect events."""
        if self._initialized:
            return
        
        log('Starting AutoReconnectController')
        self._initialized = True
        
        # Subscribe to disconnect event
        g_playerEvents.onDisconnected += self._onDisconnected
        g_playerEvents.onAccountShowGUI += self._onAccountShowGUI
        
        # Subscribe to connection manager events
        try:
            from skeletons.connection_mgr import IConnectionManager
            self._connectionMgr = dependency.instance(IConnectionManager)
            if self._connectionMgr:
                self._connectionMgr.onRejected += self._onConnectionRejected
        except Exception as e:
            log('Error subscribing to connection manager: %s' % str(e))
        
        # Patch showDisconnect to suppress error dialogs during reconnect
        self._patchShowDisconnect()
        
        log('AutoReconnectController initialized')
    
    def stop(self):
        """Stop listening and cleanup."""
        if not self._initialized:
            return
        
        log('Stopping AutoReconnectController')
        self._initialized = False
        
        # Unsubscribe from events
        g_playerEvents.onDisconnected -= self._onDisconnected
        g_playerEvents.onAccountShowGUI -= self._onAccountShowGUI
        
        # Unsubscribe from connection manager
        if self._connectionMgr:
            try:
                self._connectionMgr.onRejected -= self._onConnectionRejected
            except Exception:
                pass
            self._connectionMgr = None
        
        # Restore original showDisconnect
        self._restoreShowDisconnect()
        
        # Cancel pending reconnect
        self._cancelReconnect()
    
    def _patchShowDisconnect(self):
        """Patch showDisconnect to suppress dialog during auto-reconnect."""
        try:
            from gui import DialogsInterface
            if hasattr(DialogsInterface, 'showDisconnect'):
                self._originalShowDisconnect = DialogsInterface.showDisconnect
                
                def patchedShowDisconnect(reason=None, kickReasonType=None, expiryTime=None):
                    if self._suppressDisconnectDialog:
                        log('Suppressing disconnect dialog for auto-reconnect')
                        return
                    # Import here to get the correct default value
                    from constants import ACCOUNT_KICK_REASONS
                    if kickReasonType is None:
                        kickReasonType = ACCOUNT_KICK_REASONS.UNKNOWN
                    return self._originalShowDisconnect(reason, kickReasonType, expiryTime)
                
                DialogsInterface.showDisconnect = patchedShowDisconnect
                log('Patched showDisconnect')
        except Exception as e:
            log('Error patching showDisconnect: %s' % str(e))
    
    def _restoreShowDisconnect(self):
        """Restore original showDisconnect function."""
        try:
            if self._originalShowDisconnect is not None:
                from gui import DialogsInterface
                DialogsInterface.showDisconnect = self._originalShowDisconnect
                self._originalShowDisconnect = None
                log('Restored showDisconnect')
        except Exception as e:
            log('Error restoring showDisconnect: %s' % str(e))
    
    def _onAccountShowGUI(self, ctx):
        """Called when successfully connected and in lobby."""
        log('Successfully connected to server')
        self._suppressDisconnectDialog = False
        self._isReconnecting = False
        self._cancelReconnect()
    
    def _onConnectionRejected(self, status, responseData):
        """Called when connection attempt is rejected."""
        log('Connection rejected (status=%s), will retry in %.0f seconds' % (status, RECONNECT_INTERVAL))
        self._isReconnecting = False
        self._scheduleReconnect()
    
    def _onDisconnected(self):
        """Called when player is disconnected from server."""
        log('Disconnected from server, scheduling reconnect in %.0f seconds' % RECONNECT_INTERVAL)
        self._suppressDisconnectDialog = True
        self._scheduleReconnect()
    
    def _scheduleReconnect(self):
        """Schedule a reconnect attempt."""
        self._cancelReconnect()
        self._reconnectCallback = BigWorld.callback(RECONNECT_INTERVAL, self._doReconnect)
    
    def _cancelReconnect(self):
        """Cancel pending reconnect callback."""
        if self._reconnectCallback is not None:
            try:
                BigWorld.cancelCallback(self._reconnectCallback)
            except Exception:
                pass
            self._reconnectCallback = None
    
    def _findTargetServer(self):
        """Find the target server (RU2) from predefined hosts."""
        try:
            from predefined_hosts import g_preDefinedHosts
            
            # Get all available hosts
            hosts = g_preDefinedHosts.hosts()
            
            for host in hosts:
                # Check if server name contains our target
                if TARGET_SERVER_NAME.lower() in host.name.lower():
                    log('Found target server: %s (url: %s, peripheryID: %s)' % (
                        host.name, host.url, host.peripheryID))
                    return host
                if TARGET_SERVER_NAME.lower() in host.shortName.lower():
                    log('Found target server by short name: %s (url: %s)' % (
                        host.shortName, host.url))
                    return host
            
            log('Target server %s not found in host list' % TARGET_SERVER_NAME)
            # Log available servers for debugging
            for host in hosts:
                debug('Available server: name=%s, shortName=%s, url=%s' % (
                    host.name, host.shortName, host.url))
            
            return None
        except Exception as e:
            log('Error finding target server: %s' % str(e))
            return None
    
    def _doReconnect(self):
        """Attempt to reconnect to the target server."""
        self._reconnectCallback = None
        
        if self._isReconnecting:
            log('Reconnect already in progress, skipping')
            return
        
        log('Attempting to reconnect to %s...' % TARGET_SERVER_NAME)
        self._isReconnecting = True
        
        try:
            from skeletons.gui.login_manager import ILoginManager
            
            loginManager = dependency.instance(ILoginManager)
            
            if loginManager is None:
                log('LoginManager not available, will retry')
                self._isReconnecting = False
                self._scheduleReconnect()
                return
            
            # Find target server
            targetHost = self._findTargetServer()
            
            if targetHost is None:
                log('Could not find target server, will retry')
                self._isReconnecting = False
                self._scheduleReconnect()
                return
            
            serverUrl = targetHost.url
            
            # Try to login via WGC (Wargaming Game Center)
            if hasattr(loginManager, 'wgcAvailable') and loginManager.wgcAvailable:
                log('Attempting WGC login to %s' % serverUrl)
                loginManager.tryWgcLogin(serverUrl)
            elif hasattr(loginManager, 'lgcAvailable') and loginManager.lgcAvailable:
                log('Attempting LGC login to %s' % serverUrl)
                loginManager.tryLgcLogin(serverUrl)
            else:
                # Fallback: try to use token2 relogin if available
                log('WGC not available, trying token2 relogin')
                token2 = loginManager.getPreference('token2')
                login = loginManager.getPreference('login')
                
                if token2 and login:
                    log('Using stored credentials for relogin')
                    loginManager.initiateRelogin(login, token2, serverUrl)
                else:
                    log('No stored credentials available, cannot auto-reconnect')
                    self._suppressDisconnectDialog = False
                    self._isReconnecting = False
                    return
            
            # Reset reconnecting flag after a delay
            # (connection result will come through events)
            BigWorld.callback(5.0, self._onReconnectAttemptComplete)
            
        except Exception as e:
            log('Error during reconnect: %s' % str(e))
            self._isReconnecting = False
            self._scheduleReconnect()
    
    def _onReconnectAttemptComplete(self):
        """Called after reconnect attempt to reset state if no events fired."""
        # Only reset if we're still in reconnecting state
        # This acts as a fallback in case events don't fire
        if self._isReconnecting:
            log('Reconnect attempt timed out, scheduling retry')
            self._isReconnecting = False
            self._scheduleReconnect()


# Global auto-reconnect controller instance
g_reconnectController = AutoReconnectController()


class TrainingBotController(object):
    """
    Controller for automatically accepting training battle invites,
    setting ready status, and returning to hangar after battle starts.
    """
    hangarSpace = dependency.descriptor(IHangarSpace) # type: IHangarSpace

    def __init__(self):
        self._lastAcceptedTime = 0
        self._currentPrebattleID = None
        self._isInTrainingRoom = False
        self._invitesManager = None
        self._battleExitScheduled = False
        self._initialized = False
        self._checkReadyCallback = None
        self._chatSubscribed = False

    def start(self):
        """Start listening for events."""
        if self._initialized:
            return

        log('Starting TrainingBotController')
        self._initialized = True

        # Subscribe to player events
        g_playerEvents.onAccountShowGUI += self._onAccountShowGUI
        g_playerEvents.onAvatarReady += self._onAvatarReady
        g_playerEvents.onPrebattleJoined += self._onPrebattleJoined
        g_playerEvents.onPrebattleLeft += self._onPrebattleLeft
        self.hangarSpace.onSpaceCreate += self._onSpaceLoaded
        

        log('Subscribed to player events')

    def stop(self):
        """Stop listening for events and cleanup."""
        if not self._initialized:
            return

        log('Stopping TrainingBotController')
        self._initialized = False

        # Unsubscribe from player events
        self.hangarSpace.onSpaceCreate -= self._onSpaceLoaded
        g_playerEvents.onAccountShowGUI -= self._onAccountShowGUI
        g_playerEvents.onAvatarReady -= self._onAvatarReady
        g_playerEvents.onPrebattleJoined -= self._onPrebattleJoined
        g_playerEvents.onPrebattleLeft -= self._onPrebattleLeft

        # Unsubscribe from invites manager
        self._unsubscribeFromInvites()
        
        # Unsubscribe from chat
        self._unsubscribeFromChat()

        # Cancel any pending callbacks
        self._cancelCheckReadyCallback()

    def _onSpaceLoaded(self):
        """Called when a new space (hangar or battle) is loaded."""
        debug('Space loaded, disabling rendering')
        if not DEBUG_MODE:
            # Small delay to ensure space is fully loaded
            BigWorld.callback(0, self._disableRendering)

    def _disableRendering(self):
        """Disable 3D world rendering to save GPU resources."""
        try:
            BigWorld.worldDrawEnabled(False)
            log('3D rendering disabled')
        except Exception as e:
            log('Failed to disable 3D rendering: %s' % str(e))

    def _enableRendering(self):
        """Enable 3D world rendering."""
        try:
            BigWorld.worldDrawEnabled(True)
            log('3D rendering enabled')
        except Exception as e:
            log('Failed to enable 3D rendering: %s' % str(e))

    def _onAccountShowGUI(self, ctx):
        """Called when player enters the lobby (hangar)."""
        log('onAccountShowGUI called')
        self._subscribeToInvites()
        # Select next available vehicle after returning from battle
        BigWorld.callback(1.0, self._selectAvailableVehicle)
        # Check if we're in a training room and need to resubscribe to chat
        BigWorld.callback(1.5, self._checkAndResubscribeToChat)

    def _subscribeToInvites(self):
        """Subscribe to invite events from InvitesManager."""
        try:
            from gui.prb_control.dispatcher import g_prbLoader
            self._invitesManager = g_prbLoader.getInvitesManager()
            if self._invitesManager:
                self._invitesManager.onReceivedInviteListModified += self._onReceivedInviteListModified
                log('Subscribed to invites manager')
            else:
                log('InvitesManager not available yet, will retry')
                BigWorld.callback(1.0, self._subscribeToInvites)
        except Exception as e:
            log('Error subscribing to invites: %s' % str(e))
            BigWorld.callback(1.0, self._subscribeToInvites)

    def _unsubscribeFromInvites(self):
        """Unsubscribe from invite events."""
        if self._invitesManager:
            try:
                self._invitesManager.onReceivedInviteListModified -= self._onReceivedInviteListModified
                log('Unsubscribed from invites manager')
            except Exception:
                pass
            self._invitesManager = None

    def _onReceivedInviteListModified(self, added, changed, deleted):
        """Called when invite list is modified."""
        log('Invite list modified: added=%s, changed=%s, deleted=%s' % (added, changed, deleted))

        # Process new invites immediately
        for inviteID in added:
            self._processInvite(inviteID)
        
        # Process changed invites with a small delay (state might not be updated yet)
        for inviteID in changed:
            BigWorld.callback(0.2, lambda id=inviteID: self._processInvite(id))

    @adisp_process
    def _processInvite(self, inviteID):
        """Process a received invite."""
        if not self._invitesManager:
            return

        invite = self._invitesManager.getInvite(inviteID)
        if not invite:
            log('Invite %s not found' % inviteID)
            return

        log('Processing invite: id=%s, type=%s, peripheryID=%s, prebattleID=%s' %
              (inviteID, invite.type, invite.peripheryID, invite.prebattleID))

        # Check if invite is active
        if not invite.isActive():
            log('Ignoring inactive invite')
            return

        # Check if this is a training battle invite
        if invite.type != PREBATTLE_TYPE.TRAINING:
            log('Declining non-training invite (type=%s)' % invite.type)
            self._declineInvite(inviteID)
            return

        # Check if invite is for current server
        if invite.anotherPeriphery:
            log('Declining invite from another server (peripheryID=%s)' % invite.peripheryID)
            self._declineInvite(inviteID)
            return

        # Check if inviter is in allowed players list (if list is not empty)
        if ALLOWED_PLAYERS or ALLOWED_PLAYERS_REGEX:
            inviterName = self._getInviterName(invite)
            if inviterName:
                isAllowed = False
                
                # Check exact name matches
                if ALLOWED_PLAYERS and inviterName in ALLOWED_PLAYERS:
                    log('Accepting invite from allowed player: %s' % inviterName)
                    isAllowed = True
                
                # Check regex patterns if exact match failed
                if not isAllowed and ALLOWED_PLAYERS_REGEX:
                    for pattern in ALLOWED_PLAYERS_REGEX:
                        try:
                            if re.match(pattern, inviterName):
                                log('Accepting invite from player matching regex "%s": %s' % (pattern, inviterName))
                                isAllowed = True
                                break
                        except Exception as e:
                            log('Error matching regex pattern "%s": %s' % (pattern, str(e)))
                
                if not isAllowed:
                    log('Declining invite from non-allowed player: %s' % inviterName)
                    self._declineInvite(inviteID)
                    return
            else:
                log('Warning: Could not determine inviter name, accepting anyway')

        # Check cooldown
        currentTime = BigWorld.time()
        timeSinceLastAccept = currentTime - self._lastAcceptedTime

        if timeSinceLastAccept < INVITE_COOLDOWN:
            log('Declining invite due to cooldown (%.1f seconds remaining)' %
                  (INVITE_COOLDOWN - timeSinceLastAccept))
            self._declineInvite(inviteID)
            return

        # Check if we're currently in a training room and need to leave first
        if self._isInTrainingPrebattle():
            log('Already in training room, leaving first')
            result = yield self._doLeaveCurrentRoom()
            log('Leave result: %s' % result)
            
            # Re-check invite after leaving (it might have expired)
            invite = self._invitesManager.getInvite(inviteID)
            if not invite or not invite.isActive():
                log('Invite expired while leaving room')
                return

        # Accept the invite
        yield self._doAcceptInvite(inviteID)

    def _isInTrainingPrebattle(self):
        """Check if player is currently in a training room."""
        try:
            from gui.prb_control.dispatcher import g_prbLoader
            dispatcher = g_prbLoader.getDispatcher()
            
            if not dispatcher:
                return False
            
            entity = dispatcher.getEntity()
            if not entity:
                return False
            
            # Check if specifically in training room (type 2)
            # Hangar is type 1, we don't need to leave from hangar
            entityType = entity.getEntityType()
            if entityType == PREBATTLE_TYPE.TRAINING:
                log('Currently in training room')
                return True
            
            return False
        except Exception as e:
            log('Error checking prebattle state: %s' % str(e))
            return False

    def _getInviterName(self, invite):
        """Get the name of the player who sent the invite."""
        try:
            if hasattr(invite, 'creator') and invite.creator:
                return invite.creator
            
            log('Warning: Could not determine inviter name from invite')
            return None
            
        except Exception as e:
            log('Error getting inviter name: %s' % str(e))
            return None

    def _declineInvite(self, inviteID):
        """Decline an invite explicitly."""
        try:
            if self._invitesManager:
                self._invitesManager.declineInvite(inviteID)
                log('Declined invite %s' % inviteID)
        except Exception as e:
            log('Error declining invite: %s' % str(e))

    @adisp_async
    def _doAcceptInvite(self, inviteID, callback):
        """Accept the invite asynchronously."""
        try:
            if not self._invitesManager:
                log('Cannot accept invite: InvitesManager not available')
                callback(False)
                return

            invite = self._invitesManager.getInvite(inviteID)
            if not invite:
                log('Cannot accept invite: invite not found')
                callback(False)
                return

            if not self._invitesManager.canAcceptInvite(invite):
                log('Cannot accept invite: conditions not met')
                callback(False)
                return

            log('Accepting invite to training room (prebattleID=%s)' % invite.prebattleID)
            self._lastAcceptedTime = BigWorld.time()
            self._invitesManager.acceptInvite(inviteID)
            callback(True)

        except Exception as e:
            log('Error accepting invite: %s' % str(e))
            callback(False)

    @adisp_async
    @adisp_process
    def _doLeaveCurrentRoom(self, callback):
        """Leave the current prebattle/training room asynchronously."""
        try:
            from gui.prb_control.dispatcher import g_prbLoader
            from gui.prb_control.entities.base.ctx import LeavePrbAction
            
            dispatcher = g_prbLoader.getDispatcher()

            if not dispatcher:
                log('No dispatcher found')
                callback(True)
                return

            entity = dispatcher.getEntity()
            if not entity:
                log('No entity found')
                callback(True)
                return

            entityType = entity.getEntityType()
            log('Leaving current entity, type: %s' % entityType)
            
            # Use doLeaveAction which properly handles all entity types
            # ignoreConfirmation=True to skip confirmation dialogs
            result = yield dispatcher.doLeaveAction(LeavePrbAction(isExit=False, ignoreConfirmation=True))
            log('doLeaveAction result: %s' % result)
            callback(result)

        except Exception as e:
            log('Error leaving room: %s' % str(e))
            callback(False)

    def _onPrebattleJoined(self):
        """Called when player joins a prebattle."""
        log('onPrebattleJoined called')
        
        # Entity type might not be TRAINING yet (could be LegacyInitEntity first)
        # Schedule a delayed check to allow entity to be fully initialized
        BigWorld.callback(1.0, self._checkAndSubscribeToTrainingRoom)

    def _checkAndSubscribeToTrainingRoom(self, retryCount=0):
        """Check if we're in a training room and subscribe to chat."""
        try:
            from gui.prb_control.dispatcher import g_prbLoader
            dispatcher = g_prbLoader.getDispatcher()
            if not dispatcher:
                if retryCount < 30:
                    log('Dispatcher not available for training room check, retry %d/30' % (retryCount + 1))
                    BigWorld.callback(1, lambda: self._checkAndSubscribeToTrainingRoom(retryCount + 1))
                else:
                    log('Dispatcher not available after 30 retries, giving up')
                return
            
            entity = dispatcher.getEntity()
            if not entity:
                if retryCount < 30:
                    log('Entity not available for training room check, retry %d/30' % (retryCount + 1))
                    BigWorld.callback(1, lambda: self._checkAndSubscribeToTrainingRoom(retryCount + 1))
                else:
                    log('Entity not available after 30 retries, giving up')
                return
            
            entityType = entity.getEntityType()
            log('Checking prebattle type: %s' % entityType)
            
            if entityType != PREBATTLE_TYPE.TRAINING:
                if retryCount < 30:
                    log('Not a training room (type=%s), retry %d/30' % (entityType, retryCount + 1))
                    BigWorld.callback(1, lambda: self._checkAndSubscribeToTrainingRoom(retryCount + 1))
                else:
                    log('Not a training room (type=%s), skipping' % entityType)
                return
            
            if self._isInTrainingRoom and self._chatSubscribed:
                log('Already in training room and subscribed')
                return
            
            log('Confirmed training room, subscribing to chat')
            self._isInTrainingRoom = True
            
            # Subscribe to chat messages
            self._subscribeToChat()
            
            # Schedule setting ready status
            self._scheduleSetReady()
            
        except Exception as e:
            log('Error in training room check: %s' % str(e))

    def _onPrebattleLeft(self):
        """Called when player leaves a prebattle."""
        log('onPrebattleLeft called')
        self._isInTrainingRoom = False
        self._cancelCheckReadyCallback()
        
        # Unsubscribe from chat messages
        self._unsubscribeFromChat()

    def _scheduleSetReady(self):
        """Schedule setting ready status."""
        self._cancelCheckReadyCallback()
        self._checkReadyCallback = BigWorld.callback(1.0, self._doSetPlayerReady)

    @adisp_process
    def _doSetPlayerReady(self):
        """Set player ready status in training room using adisp."""
        try:
            from gui.prb_control.dispatcher import g_prbLoader
            from gui.prb_control.entities.base.legacy.ctx import SetPlayerStateCtx
            
            dispatcher = g_prbLoader.getDispatcher()

            if not dispatcher:
                log('Dispatcher not available')
                return

            entity = dispatcher.getEntity()
            if not entity:
                log('Entity not available')
                return

            # Check if this is a training entity
            entityType = entity.getEntityType()
            if entityType != PREBATTLE_TYPE.TRAINING:
                log('Not in training room (entityType=%s)' % entityType)
                return

            # Check if already ready
            playerInfo = entity.getPlayerInfo()
            if playerInfo and playerInfo.isReady():
                log('Player already ready')
                return

            # Set ready status using dispatcher.sendPrbRequest
            ctx = SetPlayerStateCtx(True, waitingID='prebattle/player_ready')
            result = yield dispatcher.sendPrbRequest(ctx)
            
            if result:
                log('Player set to ready')
            else:
                log('Failed to set ready, will retry')
                self._scheduleSetReady()

        except Exception as e:
            log('Error setting ready status: %s' % str(e))
            self._scheduleSetReady()

    def _cancelCheckReadyCallback(self):
        """Cancel pending check ready callback."""
        if self._checkReadyCallback is not None:
            try:
                BigWorld.cancelCallback(self._checkReadyCallback)
            except Exception:
                pass
            self._checkReadyCallback = None

    def _onAvatarReady(self):
        """Called when battle is fully loaded and ready."""
        log('onAvatarReady called - battle loaded')
        self._isInTrainingRoom = False
        self._battleExitScheduled = True

        # Exit battle now that it's fully loaded
        BigWorld.callback(0.1, self._exitBattle)

    def _exitBattle(self):
        """Exit from battle back to hangar."""
        if not self._battleExitScheduled:
            return

        self._battleExitScheduled = False

        try:
            from gui.battle_control import avatar_getter
            log('Exiting battle to return to hangar')
            avatar_getter.leaveArena()
        except Exception as e:
            log('Error exiting battle: %s' % str(e))
            # Retry if failed
            self._battleExitScheduled = True
            BigWorld.callback(1.0, self._exitBattle)

    def _selectAvailableVehicle(self):
        """Select next available vehicle that is not in battle."""
        try:
            from CurrentVehicle import g_currentVehicle
            from gui.shared.utils.requesters import REQ_CRITERIA
            from helpers import dependency
            from skeletons.gui.shared import IItemsCache

            itemsCache = dependency.instance(IItemsCache)

            # Check if current vehicle is available
            if g_currentVehicle.isPresent() and not g_currentVehicle.isInBattle():
                log('Current vehicle is available')
                return

            # Find available vehicles
            vehiclesCriteria = (REQ_CRITERIA.INVENTORY |
                               ~REQ_CRITERIA.VEHICLE.MODE_HIDDEN |
                               ~REQ_CRITERIA.VEHICLE.EVENT_BATTLE |
                               REQ_CRITERIA.VEHICLE.ACTIVE_IN_NATION_GROUP |
                               ~REQ_CRITERIA.VEHICLE.BATTLE_ROYALE |
                               REQ_CRITERIA.VEHICLE.READY)

            invVehs = itemsCache.items.getVehicles(criteria=vehiclesCriteria)

            # Filter out vehicles in battle
            availableVehs = {k: v for k, v in invVehs.iteritems() if not v.isInBattle}
            sortedByLevelVehs = sorted(availableVehs.itervalues(), key=lambda v: -v.level)

            if sortedByLevelVehs:
                # Select first available vehicle
                nextVeh = sortedByLevelVehs[0]
                log('Selecting available vehicle: %s (invID=%s)' % (nextVeh.name, nextVeh.invID))
                g_currentVehicle.selectVehicle(nextVeh.invID)
            else:
                log('No available vehicles found')

        except Exception as e:
            log('Error selecting available vehicle: %s' % str(e))

    def _subscribeToChat(self):
        """Subscribe to chat messages in training room."""
        if self._chatSubscribed:
            log('Already subscribed to chat messages')
            return
        
        try:
            from messenger.proto.events import g_messengerEvents
            g_messengerEvents.channels.onMessageReceived += self._onChatMessageReceived
            self._chatSubscribed = True
            log('Successfully subscribed to chat messages')
        except Exception as e:
            log('Error subscribing to chat: %s' % str(e))

    def _unsubscribeFromChat(self):
        """Unsubscribe from chat messages."""
        if not self._chatSubscribed:
            return
        
        try:
            from messenger.proto.events import g_messengerEvents
            g_messengerEvents.channels.onMessageReceived -= self._onChatMessageReceived
            self._chatSubscribed = False
            log('Unsubscribed from chat messages')
        except Exception as e:
            log('Error unsubscribing from chat: %s' % str(e))

    def _checkAndResubscribeToChat(self):
        """Check if we're in a training room and resubscribe to chat if needed."""
        # After returning from battle, chat subscription might be lost
        # Force resubscription by resetting the flag
        if self._chatSubscribed:
            log('Resetting chat subscription flag for resubscription')
            self._chatSubscribed = False
        
        # Use the same logic as when joining (with retry support)
        self._checkAndSubscribeToTrainingRoom(retryCount=0)

    def _onChatMessageReceived(self, message, channel):
        """Handle incoming chat message in training room."""
        try:
            # Check if we're in training room
            if not self._isInTrainingRoom:
                return
            
            # Get message text
            text = message.text if hasattr(message, 'text') else ''
            if not text:
                return
            
            # Encode to string if unicode
            if isinstance(text, types.UnicodeType):
                textStr = text.encode('utf-8')
            else:
                textStr = str(text)
            
            # Check if message is a vehicle command (starts with colon)
            textStripped = textStr.strip()
            if not textStripped.startswith(':'):
                return
            
            # Extract vehicle name query (everything after the colon)
            vehicleQuery = textStripped[1:].strip()
            if not vehicleQuery:
                log('Empty vehicle command received')
                return
            
            log('Received vehicle command: "%s"' % vehicleQuery)
            
            # Process the vehicle change command (async)
            self._processVehicleCommand(vehicleQuery)
            
        except Exception as e:
            log('Error processing chat message: %s' % str(e))

    @adisp_async
    @adisp_process
    def _setPlayerReadyState(self, isReady, callback):
        """Set player ready state (ready or not ready) asynchronously."""
        try:
            from gui.prb_control.dispatcher import g_prbLoader
            from gui.prb_control.entities.base.legacy.ctx import SetPlayerStateCtx
            
            dispatcher = g_prbLoader.getDispatcher()
            if not dispatcher:
                log('Dispatcher not available for ready state change')
                callback(False)
                return
            
            entity = dispatcher.getEntity()
            if not entity:
                log('Entity not available for ready state change')
                callback(False)
                return
            
            # Check if this is a training entity
            entityType = entity.getEntityType()
            if entityType != PREBATTLE_TYPE.TRAINING:
                log('Not in training room for ready state change')
                callback(False)
                return
            
            # Check current ready state
            playerInfo = entity.getPlayerInfo()
            if playerInfo:
                currentlyReady = playerInfo.isReady()
                if currentlyReady == isReady:
                    log('Player already in desired ready state: %s' % isReady)
                    callback(True)
                    return
            
            # Set ready status using yield
            log('Sending ready state change request: %s' % isReady)
            ctx = SetPlayerStateCtx(isReady, waitingID='prebattle/player_ready')
            result = yield dispatcher.sendPrbRequest(ctx)
            
            if result:
                log('Player ready state changed to: %s' % isReady)
            else:
                log('Failed to change ready state to: %s' % isReady)
            callback(result)
            
        except Exception as e:
            log('Error changing ready state: %s' % str(e))
            callback(False)

    @adisp_async
    def _waitForVehicleChange(self, targetInvID, callback, timeout=5.0):
        """Wait for vehicle change to complete."""
        from CurrentVehicle import g_currentVehicle
        
        startTime = BigWorld.time()
        
        def checkVehicle():
            if g_currentVehicle.isPresent() and g_currentVehicle.item.invID == targetInvID:
                callback(True)
                return
            
            if BigWorld.time() - startTime > timeout:
                log('Vehicle change timed out')
                callback(False)
                return
            
            BigWorld.callback(0.2, checkVehicle)
        
        BigWorld.callback(0.2, checkVehicle)

    @adisp_process
    def _processVehicleCommand(self, query):
        """Process vehicle change command from chat."""
        try:
            from CurrentVehicle import g_currentVehicle
            from gui.shared.utils.requesters import REQ_CRITERIA
            from skeletons.gui.shared import IItemsCache
            
            itemsCache = dependency.instance(IItemsCache)
            
            # Get all available vehicles in inventory
            vehiclesCriteria = (REQ_CRITERIA.INVENTORY |
                               ~REQ_CRITERIA.VEHICLE.MODE_HIDDEN |
                               ~REQ_CRITERIA.VEHICLE.EVENT_BATTLE |
                               REQ_CRITERIA.VEHICLE.ACTIVE_IN_NATION_GROUP |
                               ~REQ_CRITERIA.VEHICLE.BATTLE_ROYALE |
                               REQ_CRITERIA.VEHICLE.READY)
            
            invVehs = itemsCache.items.getVehicles(criteria=vehiclesCriteria)
            
            # Filter out vehicles in battle
            availableVehs = [v for v in invVehs.itervalues() if not v.isInBattle]
            
            if not availableVehs:
                log('No available vehicles to select')
                return
            
            # Find best matching vehicle
            bestMatch, score = findBestVehicleMatch(query, availableVehs)
            
            if bestMatch is None:
                log('No vehicle found matching "%s" (threshold: %.0f%%)' % (query, VEHICLE_MATCH_THRESHOLD * 100))
                return
            
            # Log match details
            log('Found vehicle match: "%s" (tag: %s) with score %.1f%% for query "%s"' % (
                safeEncode(bestMatch.userName),
                bestMatch.descriptor.name,
                score * 100,
                query
            ))
            
            # Check if this is already the current vehicle
            if g_currentVehicle.isPresent() and g_currentVehicle.item.invID == bestMatch.invID:
                log('Vehicle "%s" is already selected' % safeEncode(bestMatch.userName))
                return
            
            # Step 1: Unset ready status before changing vehicle
            log('Step 1: Unsetting ready status before vehicle change')
            unreadyResult = yield self._setPlayerReadyState(False)
            if not unreadyResult:
                log('Warning: Failed to unset ready status, continuing anyway')
            
            # Small delay to ensure state is updated
            yield self._asyncDelay(0.3)
            
            # Step 2: Select the matched vehicle
            log('Step 2: Selecting vehicle: %s (invID=%s)' % (safeEncode(bestMatch.userName), bestMatch.invID))
            g_currentVehicle.selectVehicle(bestMatch.invID)
            
            # Step 3: Wait for vehicle change to complete
            log('Step 3: Waiting for vehicle change to complete')
            vehicleChanged = yield self._waitForVehicleChange(bestMatch.invID)
            if not vehicleChanged:
                log('Warning: Vehicle change may not have completed')
            
            # Small delay to ensure UI is updated
            yield self._asyncDelay(0.5)
            
            # Step 4: Set ready status back
            log('Step 4: Setting ready status back')
            readyResult = yield self._setPlayerReadyState(True)
            if readyResult:
                log('Vehicle change complete: %s' % safeEncode(bestMatch.userName))
            else:
                log('Vehicle changed but failed to set ready status')
            
        except Exception as e:
            log('Error processing vehicle command: %s' % str(e))

    @adisp_async
    def _asyncDelay(self, delay, callback):
        """Async delay helper."""
        BigWorld.callback(delay, lambda: callback(True))


# Global controller instance
g_controller = TrainingBotController()


def init():
    log('Loaded v%s' % VERSION)
    _initModsSettings()
    log('Configuration: target_server=%s, allowed_players=%s, allowed_players_regex=%s' % (TARGET_SERVER_NAME, ALLOWED_PLAYERS, ALLOWED_PLAYERS_REGEX))
    if not ALLOWED_PLAYERS and not ALLOWED_PLAYERS_REGEX:
        log('No player restrictions: accepting invites from any player')
    else:
        restrictionCount = len(ALLOWED_PLAYERS) + len(ALLOWED_PLAYERS_REGEX)
        log('Player restrictions enabled: %d exact names, %d regex patterns' % (len(ALLOWED_PLAYERS), len(ALLOWED_PLAYERS_REGEX)))
    g_reconnectController.start()
    g_controller.start()


def fini():
    log('Unloading')
    g_controller.stop()
    g_reconnectController.stop()