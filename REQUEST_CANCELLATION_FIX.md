# Request Cancellation Fix

## Problem

When rapidly clicking tags, multiple API requests were fired simultaneously, causing:

1. **Duplicate calculations** - Server recalculated entire concept state multiple times
2. **Race conditions** - Slower request could overwrite newer state
3. **Wasted resources** - Redundant network/compute usage

### Evidence from Logs

```
Line 925: INFO: "POST /api/concepts/interact HTTP/1.1" 200 OK
...
Line 999: INFO: "POST /api/concepts/interact HTTP/1.1" 200 OK

Duplicate tag logs (every tag appears twice):
Line 935 & 937: tag_impression_impression_0_0_0 -> positive
Line 936 & 938: tag_impression_impression_0_0_1 -> positive
```

## Root Cause

The `handleTagInteraction` function had no throttling mechanism:

```javascript
// OLD: No protection against rapid clicks
const handleTagInteraction = async (tagId, preference) => {
  const response = await fetch('/api/concepts/interact', {...});
  // If user clicks again, both requests process simultaneously
};
```

## Solution: AbortController

Implemented request cancellation using the [AbortController API](https://developer.mozilla.org/en-US/docs/Web/API/AbortController):

```javascript
// NEW: Cancel previous request when new one starts
const abortControllerRef = useRef(null);

const handleTagInteraction = async (tagId, preference) => {
  // Cancel previous request if exists
  if (abortControllerRef.current) {
    abortControllerRef.current.abort();
    console.log('⏹️ Cancelled previous request');
  }

  // Create new abort controller for this request
  abortControllerRef.current = new AbortController();

  try {
    const response = await fetch('/api/concepts/interact', {
      signal: abortControllerRef.current.signal,  // Cancellable
      ...
    });
    
    // Process response...
    
  } catch (err) {
    if (err.name === 'AbortError') {
      // Request was cancelled, this is expected
      console.log('⏹️ Request aborted (newer request in progress)');
    } else {
      console.error('Error:', err);
    }
  } finally {
    abortControllerRef.current = null;
  }
};
```

## Key Changes

### 1. Added AbortController Reference

```javascript
const abortControllerRef = useRef(null);  // For cancelling in-flight requests
```

### 2. Cancel Previous Request

```javascript
if (abortControllerRef.current) {
  abortControllerRef.current.abort();
}
abortControllerRef.current = new AbortController();
```

### 3. Pass Signal to Fetch

```javascript
const response = await fetch('/api/concepts/interact', {
  signal: abortControllerRef.current.signal,  // Allow cancellation
  ...
});
```

### 4. Handle Abort Errors

```javascript
catch (err) {
  if (err.name === 'AbortError') {
    // Request was cancelled, this is expected
    console.log('⏹️ Request aborted');
  } else {
    console.error('Error:', err);
  }
}
```

### 5. Cleanup on Unmount

```javascript
useEffect(() => {
  return () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      console.log('[CLEANUP] Cancelled pending requests on unmount');
    }
  };
}, []);
```

## Benefits

✅ **No duplicate work** - Only the latest request completes
✅ **Race condition free** - Cancelled requests don't update state
✅ **Better performance** - Server processes fewer requests
✅ **Clean shutdown** - Pending requests cancelled on unmount

## Behavior

### Scenario 1: Rapid Clicking

```
User clicks Tag A → Request 1 starts
User clicks Tag B → Request 1 CANCELLED, Request 2 starts
User clicks Tag C → Request 2 CANCELLED, Request 3 starts
Request 3 completes → UI updates once
```

**Result:** Only the final click processes to completion.

### Scenario 2: Single Click

```
User clicks Tag A → Request 1 starts
Request 1 completes → UI updates
```

**Result:** Normal behavior, no cancellation.

### Scenario 3: Component Unmounts

```
User navigates away → Cleanup triggers
Pending request cancelled → No state updates attempted
```

**Result:** No memory leaks or warnings.

## Testing

Test rapid tag clicking:
1. Click multiple tags quickly (< 100ms apart)
2. Check console for "⏹️ Cancelled previous request" logs
3. Verify only ONE `[TAG INTERACTION] ✅ Updated` log per final click
4. Check server logs - should see fewer requests than before

Expected output:
```
[TAG INTERACTION] ⏹️ Cancelled previous request
[TAG INTERACTION] ⏹️ Cancelled previous request
[TAG INTERACTION] ⏹️ Request aborted (newer request in progress)
[TAG INTERACTION] ⏹️ Request aborted (newer request in progress)
[TAG INTERACTION] ✅ Updated: 17 concepts
```

## Browser Support

AbortController is supported in all modern browsers:
- ✅ Chrome 66+
- ✅ Firefox 57+
- ✅ Safari 12.1+
- ✅ Edge 16+

## Related Files

- `frontend/src/components/ConceptRefinementPanel.jsx` - Main implementation

## Next Steps

This fix addresses the duplicate calculation issue. For responsiveness concerns, we can layer **optimistic updates** on top of this cancellation mechanism:

```javascript
// Optional: Add optimistic updates while keeping cancellation
const handleTagInteraction = async (tagId, preference) => {
  // Cancel previous
  if (abortControllerRef.current) {
    abortControllerRef.current.abort();
  }
  abortControllerRef.current = new AbortController();
  
  // OPTIMISTIC: Update UI immediately
  setTagPreferences(prev => {
    const newPrefs = { ...prev };
    if (newPrefs[tagId] === preference) {
      delete newPrefs[tagId];
    } else {
      newPrefs[tagId] = preference;
    }
    return newPrefs;
  });
  
  // BACKGROUND: Sync with server (cancellable)
  try {
    const response = await fetch(..., { signal: abortControllerRef.current.signal });
    // Reconcile with server response...
  } catch (err) {
    if (err.name === 'AbortError') {
      // Cancelled, keep optimistic state
    } else {
      // Error, revert optimistic state
      setTagPreferences(serverState);
    }
  }
};
```

This would provide:
- ✅ Instant feedback (optimistic)
- ✅ No duplicate work (cancellation)
- ✅ No race conditions (latest wins)
- ✅ Server reconciliation (accuracy)

