import { useEffect, useState, useCallback } from 'react';
import { Flex, Space, Button, Slider, Typography, message, InputNumber } from 'antd';
import { FolderOpenOutlined, FileZipOutlined } from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import { invoke, formatError, logEvent } from '../lib/logger';
import type { SpriteManifest } from '../components/SpriteAnimation';
import SpriteCard from '../components/SpriteCard';
import useAppStore from '../stores/useAppStore';

const { Text } = Typography;

type SpriteEntry = SpriteManifest & { dirId: string };

export default function Sprites() {
  const { t } = useTranslation();
  const { updateSetting, settings } = useAppStore();
  const [sprites, setSprites] = useState<SpriteEntry[]>([]);
  const [spriteSrcs, setSpriteSrcs] = useState<Record<string, string>>({});
  const [selectedSpriteId, setSelectedSpriteId] = useState('');
  const [spriteSize, setSpriteSize] = useState(180);
  const [bgThreshold, setBgThreshold] = useState(0.18);
  const [bgProcessing, setBgProcessing] = useState(false);
  const loadSprites = useCallback(async () => {
    try {
      const list = await invoke<SpriteEntry[]>('list_sprites');
      setSprites(list);
      const srcs: Record<string, string> = {};
      await Promise.all(
        list.map(async (s) => {
          try {
            srcs[s.dirId] = await invoke<string>('get_sprite_image', { dirId: s.dirId, fileName: 'sprite_processed.png' });
          } catch {
            try {
              srcs[s.dirId] = await invoke<string>('get_sprite_image', { dirId: s.dirId, fileName: s.spriteFile });
            } catch { /* skip */ }
          }
        })
      );
      setSpriteSrcs(srcs);
    } catch (e) {
      message.error(formatError(e, t('settings.spriteLoadFailed')));
    }
  }, [t]);

  const loadSpriteSettings = useCallback(async () => {
    try {
      const s = await invoke<{ selected_sprite_id?: string; sprite_size?: number }>('get_settings');
      if (s.selected_sprite_id) setSelectedSpriteId(s.selected_sprite_id);
      if (s.sprite_size) setSpriteSize(s.sprite_size);
    } catch { /* ignore */ }
  }, []);

  useEffect(() => {
    loadSprites();
    loadSpriteSettings();
  }, [loadSprites, loadSpriteSettings]);

  const handleSpriteImportFolder = async () => {
    try {
      const result = await invoke<SpriteEntry | null>('import_sprite_folder');
      if (result) {
        logEvent('Sprite', 'import_folder', { name: result.name });
        message.success(t('settings.spriteImportSuccess', { name: result.name }));
        await loadSprites();
      }
    } catch (e) {
      message.error(formatError(e, t('settings.spriteImportFailed')));
    }
  };

  const handleSpriteImportZip = async () => {
    try {
      const result = await invoke<SpriteEntry | null>('import_sprite_zip');
      if (result) {
        logEvent('Sprite', 'import_zip', { name: result.name });
        message.success(t('settings.spriteImportSuccess', { name: result.name }));
        await loadSprites();
      }
    } catch (e) {
      message.error(formatError(e, t('settings.spriteImportFailed')));
    }
  };

  const handleSpriteDelete = async (dirId: string) => {
    try {
      await invoke('delete_sprite', { dirId });
      logEvent('Sprite', 'delete', { dirId });
      message.success(t('settings.spriteDeleteSuccess'));
      if (selectedSpriteId === dirId) {
        setSelectedSpriteId('');
        updateSetting('selected_sprite_id', '');
      }
      await loadSprites();
    } catch (e) {
      message.error(formatError(e, t('settings.spriteImportFailed')));
    }
  };

  const handleSpriteSelect = (dirId: string) => {
    const newValue = dirId === selectedSpriteId ? '' : dirId;
    setSelectedSpriteId(newValue);
    updateSetting('selected_sprite_id', newValue);
  };

  const handleSpriteSizeChange = (size: number) => {
    setSpriteSize(size);
    updateSetting('sprite_size', String(size));
  };

  const handleBgThresholdCommit = async (value: number) => {
    setBgThreshold(value);
    if (!selectedSpriteId) return;
    setBgProcessing(true);
    try {
      await invoke('process_sprite_background', { dirId: selectedSpriteId, threshold: value });
      logEvent('Sprite', 'process_background', { dirId: selectedSpriteId, threshold: value });
      await loadSprites();
    } catch (e) {
      message.error(formatError(e, t('settings.spriteProcessFailed')));
    }
    setBgProcessing(false);
  };


  // Sprite frame range mapping per state
  type FrameMap = Record<string, [number, number]>;
  type AllFrameMaps = Record<string, FrameMap>;
  const [frameMaps, setFrameMaps] = useState<AllFrameMaps>(() => {
    try { return JSON.parse((settings as Record<string,unknown>).sprite_frame_maps as string || '{}'); }
    catch { return {}; }
  });

  const handleFrameChange = async (state: string, startOrEnd: 'start' | 'end', value: number | null) => {
    if (!selectedSpriteId || value == null) return;
    const current = frameMaps[selectedSpriteId] || {};
    const range: [number, number] = [...(current[state] || [0, 0])] as [number, number];
    range[startOrEnd === 'start' ? 0 : 1] = value;
    const updated = { ...frameMaps, [selectedSpriteId]: { ...current, [state]: range } };
    setFrameMaps(updated);
    await updateSetting('sprite_frame_maps', JSON.stringify(updated));
  };

  const selectedFrames = selectedSpriteId ? (frameMaps[selectedSpriteId] || {}) : {};
  const stateLabels: [string, string][] = [
    ['idle', '空闲'], ['recording', '录音'], ['filtering', '过滤'],
    ['processing', '处理'], ['transcribing', '转录'], ['pasting', '粘贴'],
  ];
  return (
    <Flex vertical gap="large" style={{ width: '100%' }}>
      <Typography.Title level={3}>{t('sprites.title')}</Typography.Title>
      <Flex vertical gap={12} style={{ width: '100%' }}>
        <Space>
          <Button icon={<FolderOpenOutlined />} onClick={handleSpriteImportFolder}>
            {t('settings.spriteImportFolder')}
          </Button>
          <Button icon={<FileZipOutlined />} onClick={handleSpriteImportZip}>
            {t('settings.spriteImportZip')}
          </Button>
        </Space>

        {sprites.length === 0 ? (
          <Text type="secondary">{t('settings.spriteNoData')}</Text>
        ) : (
          <Flex wrap="wrap" gap={12}>
            {sprites.map((sprite) =>
              spriteSrcs[sprite.dirId] && (
                <SpriteCard
                  key={sprite.dirId}
                  manifest={sprite}
                  imageSrc={spriteSrcs[sprite.dirId]}
                  isSelected={selectedSpriteId === sprite.dirId}
                  onSelect={() => handleSpriteSelect(sprite.dirId)}
                  onDelete={() => handleSpriteDelete(sprite.dirId)}
                />
              )
            )}
          </Flex>
        )}

        {sprites.length > 0 && (
          <Flex vertical gap={16} style={{ maxWidth: 400 }}>
            <div>
              <Flex justify="space-between" align="center">
                <Text>{t('settings.spriteSize')}</Text>
                <Text type="secondary">{spriteSize}px</Text>
              </Flex>
              <Slider min={80} max={300} step={10} value={spriteSize} onChange={handleSpriteSizeChange} />
            </div>
            <div>
              <Flex justify="space-between" align="center">
                <Text>{t('settings.spriteBgRemoval')}</Text>
                <Text type="secondary">{bgThreshold.toFixed(2)}</Text>
              </Flex>
              <Slider
                min={0.01} max={0.50} step={0.01}
                value={bgThreshold}
                onChange={setBgThreshold}
                onChangeComplete={handleBgThresholdCommit}
                disabled={!selectedSpriteId || bgProcessing}
              />
              <Text type="secondary" style={{ fontSize: 12 }}>
                {t('settings.spriteBgRemovalHint')}
              </Text>
            </div>

            {selectedSpriteId && (
              <div>
                <Text strong style={{ display: 'block', marginBottom: 8 }}>帧映射 (Frame Map)</Text>
                {stateLabels.map(([key, label]) => {
                  const frames = selectedFrames[key] || [0, 0];
                  return (
                    <Flex key={key} align="center" gap={8} style={{ marginBottom: 4 }}>
                      <Text style={{ minWidth: 48 }}>{label}</Text>
                      <InputNumber
                        size="small" min={0} max={99}
                        value={frames[0]} style={{ width: 56 }}
                        onChange={(v) => handleFrameChange(key, 'start', v)}
                      />
                      <Text type="secondary">–</Text>
                      <InputNumber
                        size="small" min={0} max={99}
                        value={frames[1]} style={{ width: 56 }}
                        onChange={(v) => handleFrameChange(key, 'end', v)}
                      />
                    </Flex>
                  );
                })}
              </div>
            )}

            {/* Frame preview strip */}
            {selectedSpriteId && spriteSrcs[selectedSpriteId] && (() => {
              const sprite = sprites.find(s => s.dirId === selectedSpriteId);
              if (!sprite) return null;
              const frameSize = 36;
              const gap = 4;
              return (
                <div style={{ marginTop: 8 }}>
                  <Text strong style={{ display: 'block', marginBottom: 4 }}>帧预览</Text>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap }}>
                    {Array.from({ length: sprite.frameCount }, (_, i) => (
                      <div key={i} style={{ textAlign: 'center' }}>
                        <canvas
                          ref={(el) => {
                            if (!el) return;
                            const img = new Image();
                            img.onload = () => {
                              const ctx = el.getContext('2d');
                              if (!ctx) return;
                              const col = i % sprite.columns;
                              const row = Math.floor(i / sprite.columns);
                              el.width = frameSize;
                              el.height = frameSize;
                              ctx.imageSmoothingEnabled = false;
                              ctx.drawImage(
                                img,
                                col * sprite.frameWidth, row * sprite.frameHeight,
                                sprite.frameWidth, sprite.frameHeight,
                                0, 0, frameSize, frameSize,
                              );
                            };
                            img.src = spriteSrcs[selectedSpriteId];
                          }}
                          style={{ border: '1px solid #d9d9d9', borderRadius: 4 }}
                        />
                        <Text type="secondary" style={{ fontSize: 10, display: 'block' }}>{i}</Text>
                      </div>
                    ))}
                  </div>
                </div>
              );
            })()}
          </Flex>
        )}
      </Flex>
    </Flex>
  );
}
