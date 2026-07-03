import { describe, test, expect, vi, beforeEach } from 'vitest';

// Mocks must be hoisted before the store import.
const invokeMock = vi.fn();
const broadcastMock = vi.fn();

vi.mock('../lib/logger', () => ({
  invoke: (...args: unknown[]) => invokeMock(...args),
  formatError: (e: unknown) => String(e),
  logEvent: vi.fn(),
}));
vi.mock('../lib/broadcast', () => ({
  broadcast: (...args: unknown[]) => broadcastMock(...args),
  onBroadcast: () => () => {},
}));

import useAppStore from '../stores/useAppStore';

describe('useAppStore.updateSetting', () => {
  beforeEach(() => {
    invokeMock.mockReset();
    broadcastMock.mockReset();
    invokeMock.mockResolvedValue(undefined);
  });

  test('broadcasts settings-changed with the key after persisting', async () => {
    // The recorder float window listens for this broadcast to reload its
    // sprite. Without it, switching sprites in settings never reaches the
    // floating window until the app restarts.
    await useAppStore.getState().updateSetting('selected_sprite_id', 'ABC');

    expect(invokeMock).toHaveBeenCalledWith('update_setting', {
      key: 'selected_sprite_id',
      value: 'ABC',
    });
    expect(broadcastMock).toHaveBeenCalledWith('settings-changed', 'selected_sprite_id');
  });

  test('broadcasts for sprite_size changes too', async () => {
    await useAppStore.getState().updateSetting('sprite_size', '200');

    expect(broadcastMock).toHaveBeenCalledWith('settings-changed', 'sprite_size');
  });

  test('does not broadcast when backend update fails', async () => {
    invokeMock.mockRejectedValueOnce(new Error('boom'));

    await expect(
      useAppStore.getState().updateSetting('selected_sprite_id', 'X'),
    ).rejects.toThrow('boom');

    expect(broadcastMock).not.toHaveBeenCalled();
  });
});
