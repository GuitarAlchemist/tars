import { useQuery } from '@tanstack/react-query';
import { mockTarsApi } from '../services/tarsApi';

export const useSystemStatus = () => {
  return useQuery({
    queryKey: ['systemStatus'],
    queryFn: mockTarsApi.getSystemStatus,
    refetchInterval: 5000, // Refetch every 5 seconds
  });
};
