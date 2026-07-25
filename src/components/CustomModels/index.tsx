import { useCallback, useState } from 'react';
import { Button, Col, Dropdown, Empty, Row, Space, Spin, message } from 'antd';
import { FolderOpenOutlined, ImportOutlined, DownOutlined, PlusOutlined } from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import { formatError } from '../../lib/logger';
import useAppStore from '../../stores/useAppStore';
import { useCustomModels } from './useCustomModels';
import { getCustomBridge } from './bridge';
import { CustomModelCard } from './CustomModelCard';
import { CreateModelDialog } from './CreateModelDialog';

/**
 * Settings page section for custom YAML-defined models.
 *
 * Aggregates list scan into per-item status (via useCustomModels) and
 * delegates rendering to <CustomModelCard /> for the four states.
 * Surface scan errors via toast — silent failures hide misconfigured
 * YAML files from the user.
 */
export function CustomModelsSection() {
  const { t } = useTranslation();
  const selectedModelId = useAppStore((s) => s.settings.selected_model_id);
  const [importing, setImporting] = useState(false);
  const [createOpen, setCreateOpen] = useState(false);
  const [editSpecPath, setEditSpecPath] = useState<string | null>(null);
  const [examples, setExamples] = useState<Array<{ key: string; label: string }>>([]);

  const reportScanError = useCallback(
    (err: unknown) => {
      const msg = formatError(err, 'unknown error');
      message.error(t('customModels.scanFailed', { msg }));
    },
    [t],
  );

  const { items, loading, refresh } = useCustomModels({ onError: reportScanError });


  const handleEdit = useCallback((specPath: string) => {
    setEditSpecPath(specPath);
    setCreateOpen(true);
  }, []);
  const safeRefresh = useCallback(() => {
    refresh().catch(reportScanError);
  }, [refresh, reportScanError]);

  const handleImportExample = async (fileName: string) => {
    setImporting(true);
    try {
      await getCustomBridge().invoke('custom-import-example', fileName);
      message.success(t('customModels.importSuccess'));
      safeRefresh();
    } catch (e) {
      message.error(formatError(e, t('customModels.importFailed')));
    } finally {
      setImporting(false);
    }
  };

  const handleOpenFolder = async () => {
    try {
      await getCustomBridge().invoke('custom-open-dir');
    } catch (e) {
      message.error(formatError(e, t('customModels.openFolderFailed')));
    }
  };

  // Build dropdown items from available examples on click
  const handleMenuClick = async ({ key }: { key: string }) => {
    await handleImportExample(key);
  };
  const menuProps = {
    items: examples,
    onClick: handleMenuClick,
  };

  // Load examples lazily when the dropdown opens
  const handleDropdownVisibleChange = async (open: boolean) => {
    if (open) {
      try {
        const result = (await getCustomBridge().invoke('custom-list-examples')) as Array<{
          name: string;
          isDir: boolean;
        }>;
        setExamples(result.map((ex) => ({ key: ex.name, label: ex.name })));
      } catch {
        // ignore — menu stays empty
      }
    }
  };

  return (
    <section data-testid="custom-models-section">
      <Space
        style={{
          width: '100%',
          justifyContent: 'flex-end',
          marginBottom: 12,
        }}
      >
        <Button type="primary" icon={<PlusOutlined />} onClick={() => { setEditSpecPath(null); setCreateOpen(true); }}>
          {t('customModels.createModel')}
        </Button>
        <Dropdown menu={menuProps} onOpenChange={handleDropdownVisibleChange}>
          <Button icon={<ImportOutlined />} loading={importing}>
            {t('customModels.importExample')} <DownOutlined />
          </Button>
        </Dropdown>
        <Button icon={<FolderOpenOutlined />} onClick={handleOpenFolder}>
          {t('customModels.openFolder')}
        </Button>
      </Space>

      {loading ? (
        <Spin />
      ) : items.length === 0 ? (
        <Empty description={t('customModels.empty')} />
      ) : (
        <Row gutter={[16, 16]}>
          {items.map((item) => {
            const key = item.kind === 'invalid' ? item.sourcePath : item.spec.id;
            const isActive =
              item.kind !== 'invalid' && selectedModelId === item.spec.id;
            return (
              <Col xs={24} sm={12} md={8} key={key}>
                <CustomModelCard
                  status={item}
                  isActive={isActive}
                  onChange={safeRefresh}
                  onEdit={handleEdit}
                />
              </Col>
            );
          })}
        </Row>
      )}
      <CreateModelDialog
        open={createOpen}
        editSpecPath={editSpecPath}
        onClose={() => { setCreateOpen(false); setEditSpecPath(null); }}
        onCreated={safeRefresh}
      />
    </section>
  );
}

export default CustomModelsSection;
